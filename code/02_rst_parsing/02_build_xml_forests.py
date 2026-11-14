
# ============================================================
# 02 — Build, load and validate grouped XML forests
# ============================================================

import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# RST tree conversion
# ------------------------------------------------------------

def attributes(node):
    if not isinstance(node, dict):
        raise TypeError(
            f"Unexpected RST node type: {type(node)}"
        )

    attrs = node.get("__attributes__", node)

    if not isinstance(attrs, dict):
        raise TypeError(
            f"Unexpected RST attributes: {type(attrs)}"
        )

    return attrs


def safe_label(value):
    value = str(value)

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )

    return value.strip("._-") or "UNKNOWN"


def convert_rst(node):
    """
    Convert an RST tree into XML.

    Internal nodes:
        R_<relation>_<nuclearity>

    Leaves:
        EDU

    Child order is preserved.
    """

    if isinstance(node, list):
        if len(node) != 1:
            raise ValueError(
                f"Expected one RST node, found {len(node)}"
            )
        return convert_rst(node[0])

    attrs = attributes(node)

    left = attrs.get("left")
    right = attrs.get("right")

    if left is None and right is None:
        return ET.Element("EDU")

    if left is None or right is None:
        raise ValueError(
            "RST node has only one child"
        )

    relation = safe_label(
        attrs.get("relation", "UNKNOWN")
    )

    nuclearity = safe_label(
        attrs.get("nuclearity", "UNKNOWN")
    )

    element = ET.Element(
        f"R_{relation}_{nuclearity}"
    )

    element.append(convert_rst(left))
    element.append(convert_rst(right))

    return element


def get_rst_root(record):
    result = record["rst_result"]
    roots = result["rst"]

    if isinstance(roots, list):
        if len(roots) != 1:
            raise ValueError(
                f"Document {record['id']} has "
                f"{len(roots)} RST roots"
            )
        return roots[0]

    return roots


# ------------------------------------------------------------
# Load parsed RST records
# ------------------------------------------------------------

def load_rst_records(trees_file):
    trees_file = Path(trees_file)

    records = {}

    with trees_file.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
                doc_id = str(record["id"])

            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid RST record at line {line_number}"
                ) from exc

            if doc_id in records:
                raise ValueError(
                    f"Duplicate RST ID: {doc_id}"
                )

            records[doc_id] = record

    if not records:
        raise ValueError(
            f"No RST records found in {trees_file}"
        )

    return records


# ------------------------------------------------------------
# Automatic genre extraction
# ------------------------------------------------------------

def extract_genre(trees_file):
    """
    Example directory:
        domain-abstracts__prompting-all__...

    Returns:
        abstracts
    """

    trees_file = Path(trees_file)

    run_name = trees_file.parent.parent.name

    match = re.search(
        r"(?:^|__)domain-(.*?)(?:__|$)",
        run_name,
    )

    if not match:
        raise ValueError(
            "Cannot extract genre from directory: "
            f"{run_name}"
        )

    genre = match.group(1)

    if not genre:
        raise ValueError(
            "Extracted genre is empty"
        )

    return genre


# ------------------------------------------------------------
# Optional external metadata
# ------------------------------------------------------------

def load_metadata(metadata_file):
    path = Path(metadata_file)

    if path.suffix.lower() == ".csv":
        metadata = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

    elif path.suffix.lower() == ".jsonl":
        rows = []

        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        metadata = pd.DataFrame(rows).fillna("")
        metadata = metadata.astype(str)

    else:
        raise ValueError(
            "Metadata must be CSV or JSONL"
        )

    if "id" not in metadata.columns:
        raise ValueError(
            "Metadata must contain an 'id' column"
        )

    metadata["id"] = metadata["id"].astype(str)

    if metadata["id"].duplicated().any():
        raise ValueError(
            "Duplicate metadata IDs"
        )

    return metadata


# ------------------------------------------------------------
# Prepare document metadata
# ------------------------------------------------------------

def prepare_metadata(
    trees_file,
    records,
    metadata_file=None,
):
    genre = extract_genre(trees_file)

    metadata = pd.DataFrame([
        {
            "id": doc_id,
            "genre": genre,
            "source": str(record["source"]),
            "n_sentences": str(record["n_sentences"]),
        }
        for doc_id, record in records.items()
    ])

    if metadata_file is None:
        return metadata

    external = load_metadata(metadata_file)

    if set(external["id"]) != set(metadata["id"]):
        raise ValueError(
            "RST/metadata ID mismatch"
        )

    overlapping = [
        column
        for column in external.columns
        if column in metadata.columns
        and column != "id"
    ]

    comparison = metadata.merge(
        external,
        on="id",
        suffixes=("_rst", "_external"),
        validate="one_to_one",
    )

    for column in overlapping:
        rst_values = comparison[
            f"{column}_rst"
        ].astype(str)

        external_values = comparison[
            f"{column}_external"
        ].astype(str)

        if not rst_values.equals(external_values):
            raise ValueError(
                f"Conflicting metadata: {column}"
            )

    additional_columns = [
        column
        for column in external.columns
        if column not in metadata.columns
    ]

    if additional_columns:
        metadata = metadata.merge(
            external[["id"] + additional_columns],
            on="id",
            validate="one_to_one",
        )

    return metadata


# ------------------------------------------------------------
# Validate metadata and grouping
# ------------------------------------------------------------

def validate_metadata(
    metadata,
    group_by,
    records,
):
    if not group_by:
        raise ValueError(
            "At least one grouping variable is required"
        )

    if len(group_by) != len(set(group_by)):
        raise ValueError(
            "Duplicate grouping variables"
        )

    missing = [
        column
        for column in group_by
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            f"Missing grouping columns: {missing}"
        )

    if set(metadata["id"]) != set(records):
        raise ValueError(
            "Metadata IDs differ from RST IDs"
        )

    for column in group_by:
        if metadata[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"Empty grouping values in {column}"
            )


def group_documents(metadata, group_by):
    groups = defaultdict(list)

    for _, row in metadata.iterrows():
        key = tuple(
            str(row[column])
            for column in group_by
        )

        groups[key].append(str(row["id"]))

    return dict(groups)


# ------------------------------------------------------------
# Build grouped XML forests
# ------------------------------------------------------------

def build_forests(
    trees_file,
    metadata_file=None,
    group_by=None,
):
    """
    Build grouped XML forests.

    Returns the output directory.

    Existing forests are replaced only after all new
    XML files have been generated and validated.
    """

    if group_by is None:
        group_by = [
            "genre",
            "source",
            "n_sentences",
        ]

    trees_file = Path(trees_file).resolve()

    if not trees_file.is_file():
        raise FileNotFoundError(trees_file)

    output_dir = trees_file.parent / "forests"

    records = load_rst_records(trees_file)

    metadata = prepare_metadata(
        trees_file,
        records,
        metadata_file,
    )

    validate_metadata(
        metadata,
        group_by,
        records,
    )

    groups = group_documents(
        metadata,
        group_by,
    )

    converted = {}

    for doc_id, record in records.items():
        try:
            converted[doc_id] = convert_rst(
                get_rst_root(record)
            )

        except Exception as exc:
            raise ValueError(
                f"RST conversion failed for {doc_id}"
            ) from exc

    assigned_ids = [
        doc_id
        for ids in groups.values()
        for doc_id in ids
    ]

    counts = Counter(assigned_ids)

    if (
        set(assigned_ids) != set(records)
        or len(assigned_ids) != len(records)
        or any(count != 1 for count in counts.values())
    ):
        raise RuntimeError(
            "Invalid document partition"
        )

    # Build in a temporary sibling directory
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix="forests_staging_",
            dir=trees_file.parent,
        )
    )

    manifest = {
        "trees_file": str(trees_file),
        "metadata_file": (
            str(Path(metadata_file).resolve())
            if metadata_file is not None
            else None
        ),
        "group_by": list(group_by),
        "total_trees": len(records),
        "total_forests": len(groups),
        "forests": [],
    }

    try:
        for index, (key, ids) in enumerate(
            sorted(groups.items()),
            start=1,
        ):
            group_info = dict(zip(group_by, key))

            label = "__".join(
                f"{safe_label(column)}-{safe_label(value)}"
                for column, value in group_info.items()
            )

            filename = f"{index:04d}__{label}.xml"

            forest = ET.Element("forest")
            document_tags = {}

            for doc_index, doc_id in enumerate(
                ids,
                start=1,
            ):
                doc_tag = f"DOC_{doc_index:06d}"

                wrapper = ET.SubElement(
                    forest,
                    doc_tag,
                )

                wrapper.append(converted[doc_id])

                document_tags[doc_tag] = doc_id

            path = staging_dir / filename

            ET.ElementTree(forest).write(
                path,
                encoding="utf-8",
                xml_declaration=True,
            )

            parsed = ET.parse(path).getroot()

            if parsed.tag != "forest":
                raise ValueError(
                    f"Invalid XML root: {filename}"
                )

            if len(parsed) != len(ids):
                raise ValueError(
                    f"Invalid document count: {filename}"
                )

            if [doc.tag for doc in parsed] != list(document_tags):
                raise ValueError(
                    f"Invalid document tags: {filename}"
                )

            if any(len(doc) != 1 for doc in parsed):
                raise ValueError(
                    f"Invalid document wrapper: {filename}"
                )

            manifest["forests"].append({
                "file": filename,
                "group": group_info,
                "n_trees": len(ids),
                "ids": ids,
                "document_tags": document_tags,
            })

            print(
                f"[{index}/{len(groups)}] "
                f"{filename}: {len(ids)} trees"
            )

        manifest_path = (
            staging_dir / "forests_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Publish only after successful generation
        if output_dir.exists():
            backup_dir = Path(
                tempfile.mkdtemp(
                    prefix="forests_previous_",
                    dir=trees_file.parent,
                )
            )
            backup_dir.rmdir()

            output_dir.rename(backup_dir)

            try:
                staging_dir.rename(output_dir)
            except Exception:
                backup_dir.rename(output_dir)
                raise

            shutil.rmtree(backup_dir)

        else:
            staging_dir.rename(output_dir)

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print("\nFOREST GENERATION COMPLETE")
    print("=" * 55)
    print(f"RST trees: {len(records)}")
    print(f"XML forests: {len(groups)}")
    print(f"Output: {output_dir}")

    return output_dir


# ------------------------------------------------------------
# Load and validate existing XML forests
# ------------------------------------------------------------

def load_existing_forests(
    trees_file,
    group_by=None,
    metadata_file=None,
    validate=True,
):
    """
    Return the manifest when existing forests are valid.

    Return None when no forest manifest exists.

    Raise an error when existing forests are invalid.
    """

    if group_by is None:
        group_by = [
            "genre",
            "source",
            "n_sentences",
        ]

    trees_file = Path(trees_file).resolve()
    forests_dir = trees_file.parent / "forests"
    manifest_file = forests_dir / "forests_manifest.json"

    if not manifest_file.exists():
        if forests_dir.exists() and any(forests_dir.iterdir()):
            raise RuntimeError(
                "Forest directory exists without a manifest. "
                "Use rebuild=True to regenerate."
            )
        return None

    manifest = json.loads(
        manifest_file.read_text(encoding="utf-8")
    )

    if manifest.get("group_by") != list(group_by):
        raise ValueError(
            "Existing forests use different grouping variables. "
            "Use rebuild=True."
        )

    if (
        Path(manifest["trees_file"]).resolve()
        != trees_file
    ):
        raise ValueError(
            "Existing forests belong to a different dataset."
        )

    expected_metadata = (
        str(Path(metadata_file).resolve())
        if metadata_file is not None
        else None
    )

    if manifest.get("metadata_file") != expected_metadata:
        raise ValueError(
            "External metadata configuration has changed. "
            "Use rebuild=True."
        )

    entries = manifest.get("forests", [])

    if (
        not entries
        or len(entries) != manifest.get("total_forests")
    ):
        raise ValueError(
            "Invalid forest manifest"
        )

    filenames = [entry["file"] for entry in entries]

    if len(set(filenames)) != len(filenames):
        raise ValueError(
            "Duplicate forest filenames"
        )

    for filename in filenames:
        if not (forests_dir / filename).is_file():
            raise FileNotFoundError(
                forests_dir / filename
            )

    if validate:
        records = load_rst_records(trees_file)

        metadata = prepare_metadata(
            trees_file,
            records,
            metadata_file,
        )

        validate_metadata(
            metadata,
            group_by,
            records,
        )

        expected_groups = group_documents(
            metadata,
            group_by,
        )

        actual_groups = {}
        assigned_ids = []

        for entry in entries:
            path = forests_dir / entry["file"]

            root = ET.parse(path).getroot()

            if root.tag != "forest":
                raise ValueError(
                    f"Invalid root: {path.name}"
                )

            mapping = entry["document_tags"]
            ids = entry["ids"]

            if len(root) != entry["n_trees"]:
                raise ValueError(
                    f"Tree count mismatch: {path.name}"
                )

            if [doc.tag for doc in root] != list(mapping):
                raise ValueError(
                    f"Document tag mismatch: {path.name}"
                )

            if list(mapping.values()) != ids:
                raise ValueError(
                    f"Document ID mismatch: {path.name}"
                )

            if any(len(doc) != 1 for doc in root):
                raise ValueError(
                    f"Invalid wrapper: {path.name}"
                )

            key = tuple(
                str(entry["group"][column])
                for column in group_by
            )

            if key in actual_groups:
                raise ValueError(
                    f"Duplicate forest group: {key}"
                )

            actual_groups[key] = ids
            assigned_ids.extend(ids)

        if (
            len(assigned_ids) != len(records)
            or set(assigned_ids) != set(records)
            or any(
                count != 1
                for count in Counter(assigned_ids).values()
            )
        ):
            raise ValueError(
                "Invalid document assignments"
            )

        if {
            key: set(ids)
            for key, ids in actual_groups.items()
        } != {
            key: set(ids)
            for key, ids in expected_groups.items()
        }:
            raise ValueError(
                "Forest grouping differs from current metadata"
            )

    return manifest


# ------------------------------------------------------------
# Load or build XML forests
# ------------------------------------------------------------

def load_or_build_forests(
    trees_file,
    metadata_file=None,
    group_by=None,
    rebuild=False,
    validate=True,
):
    """
    Main entry point for notebook orchestration.

    If forests exist:
        load and validate them.

    If forests do not exist:
        build them.

    If rebuild=True:
        regenerate all forests.

    Returns:
        forests_dir
        forest_paths
        forest_index
        manifest
    """

    if group_by is None:
        group_by = [
            "genre",
            "source",
            "n_sentences",
        ]

    trees_file = Path(trees_file).resolve()
    forests_dir = trees_file.parent / "forests"

    if not trees_file.is_file():
        raise FileNotFoundError(trees_file)

    if rebuild:
        print("Rebuilding XML forests...")

        build_forests(
            trees_file=trees_file,
            metadata_file=metadata_file,
            group_by=group_by,
        )

    manifest = load_existing_forests(
        trees_file=trees_file,
        group_by=group_by,
        metadata_file=metadata_file,
        validate=validate,
    )

    if manifest is None:
        print("No existing forests found. Building...")

        build_forests(
            trees_file=trees_file,
            metadata_file=metadata_file,
            group_by=group_by,
        )

        manifest = load_existing_forests(
            trees_file=trees_file,
            group_by=group_by,
            metadata_file=metadata_file,
            validate=True,
        )

    forest_paths = [
        forests_dir / entry["file"]
        for entry in manifest["forests"]
    ]

    forest_index = {
        tuple(
            str(entry["group"][column])
            for column in group_by
        ): forests_dir / entry["file"]
        for entry in manifest["forests"]
    }

    print("\nXML FORESTS READY")
    print("=" * 55)
    print(f"Forests: {manifest['total_forests']}")
    print(f"RST trees: {manifest['total_trees']}")
    print(f"Directory: {forests_dir}")

    return (
        forests_dir,
        forest_paths,
        forest_index,
        manifest,
    )
