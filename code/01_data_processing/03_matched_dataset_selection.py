"""RAID matched-source dataset selection, full-text recovery, and diagnostics.

Place this module alongside the existing RAID sentence-analysis script under
``code/01_.../``. It uses ``code/config.py`` and the existing ``documents`` and
``sentences`` DataFrames produced by ``analyze_sentence_lengths``.

Matching is strict: one document per (source_id, source), every configured
source present, and at least ``k`` sentences in *every* document. ``m=None``
retains all eligible source_id groups; otherwise exactly ``m`` complete groups
are sampled without replacement. Original complete texts are recovered from
RAID's source CSV by document ID; they are never reconstructed from sentences.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (  # noqa: E402
    CHUNK_SIZE,
    DOMAIN_NAMES,
    MODEL_NAMES,
    MODEL_ORDER,
    RANDOM_SEED,
    locate_training_file,
)

SELECTION_VERSION = 1


def _slug(value: object) -> str:
    if value is None:
        return "all"
    if isinstance(value, (list, tuple, set)):
        value = "-".join(sorted(map(str, value)))
    return re.sub(r"[^a-z0-9_-]+", "-", str(value).strip().lower()).strip("-") or "empty"


def _canonical_domain(domain: str) -> str:
    for key, label in DOMAIN_NAMES.items():
        if str(domain).strip().lower() in {str(key).lower(), str(label).lower()}:
            return str(key)
    raise ValueError(f"Unknown RAID genre/domain: {domain!r}")


def _find(columns, *aliases):
    lookup = {str(col).strip().lower(): col for col in columns}
    return next((lookup[a.lower()] for a in aliases if a.lower() in lookup), None)


def _matches(series: pd.Series, value) -> pd.Series:
    if value is None:
        return pd.Series(True, index=series.index)
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    if not values:
        raise ValueError("An empty filter is not allowed")
    allowed = {str(v).strip().lower() for v in values}
    normalized = series.astype("string").str.strip().str.lower()
    if allowed == {"null"}:
        return (normalized.isna() | normalized.isin(["", "none", "null"])).fillna(False)
    return normalized.isin(allowed).fillna(False)


def _output_directory(root: Path, filters: dict, k: int, m: int | None, seed: int) -> Path:
    parts = [
        "matched",
        f"genre-{_slug(filters['domain'])}",
        f"prompting-{_slug(filters['prompting'])}",
        f"attack-{_slug(filters['attack'])}",
        f"decoding-{_slug(filters['decoding'])}",
        f"repetition_penalty-{_slug(filters['repetition_penalty'])}",
        f"k-{k}",
        "all" if m is None else f"m-{m}__seed-{seed}",
    ]
    result = root
    for part in parts:
        result /= part
    return result


def _resolve_filtered_root(output_root: str | Path | None) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    # CODE_DIR is <repository>/code, so the default is <repository>/data/filtered.
    return CODE_DIR.parent / "data" / "filtered"


def _validate_statistics(documents: pd.DataFrame, sentences: pd.DataFrame) -> None:
    required_docs = {"id", "source_id", "source", "n_sentences", "n_words"}
    required_sentences = {"id", "sentence_idx", "n_words"}
    if missing := required_docs - set(documents.columns):
        raise ValueError(f"Missing document columns: {sorted(missing)}")
    if missing := required_sentences - set(sentences.columns):
        raise ValueError(f"Missing sentence columns: {sorted(missing)}")
    if documents["id"].isna().any() or documents["id"].astype("string").duplicated().any():
        raise ValueError("Document IDs must be non-null and unique")
    if sentences[["id", "sentence_idx"]].isna().any() or sentences.duplicated(["id", "sentence_idx"]).any():
        raise ValueError("Sentence IDs and indices must be non-null and unique")
    if documents[["n_sentences", "n_words"]].isna().any().any():
        raise ValueError("Missing document statistics")


def _recover_full_texts(
    ids: set[str], source_by_id: dict[str, str], source_csv: Path,
    filters: dict, *, chunksize: int,
) -> dict[str, str]:
    """Read RAID in chunks; retrieve only the selected IDs and verify sources.

    Filters are verified against the selected statistics metadata upstream;
    matching by unique ID plus source protects against accidental wrong joins.
    """
    if not source_csv.is_file():
        raise FileNotFoundError(f"RAID source CSV not found: {source_csv}")
    recovered: dict[str, str] = {}
    seen: set[str] = set()
    for chunk in pd.read_csv(source_csv, chunksize=chunksize, dtype="string"):
        id_col = _find(chunk.columns, "id")
        source_col = _find(chunk.columns, "model", "source")
        text_col = _find(chunk.columns, "generation", "text", "response")
        if any(col is None for col in (id_col, source_col, text_col)):
            raise ValueError("RAID source CSV requires id, model/source, generation/text/response")
        selected = chunk.loc[chunk[id_col].isin(ids), [id_col, source_col, text_col]]
        for doc_id, raw_source, raw_text in selected.itertuples(index=False, name=None):
            doc_id = str(doc_id)
            if doc_id in seen:
                raise ValueError(f"Duplicate document ID in original RAID CSV: {doc_id}")
            seen.add(doc_id)
            raw_source_normalized = str(raw_source).strip().lower()
            canonical_source = MODEL_NAMES.get(raw_source_normalized)
            if canonical_source is None and str(raw_source) in source_by_id.values():
                canonical_source = str(raw_source)
            if canonical_source != source_by_id[doc_id]:
                raise ValueError(
                    f"Source mismatch for id={doc_id}: expected {source_by_id[doc_id]!r}, "
                    f"found {raw_source!r}"
                )
            if pd.isna(raw_text) or not str(raw_text).strip():
                raise ValueError(f"Missing original full text for document {doc_id}")
            recovered[doc_id] = str(raw_text)
        if len(recovered) == len(ids):
            break
    if len(recovered) != len(ids):
        missing = sorted(ids - recovered.keys())
        raise ValueError(f"Could not recover {len(missing)} original documents; first IDs: {missing[:10]}")
    return recovered


def _source_summary(documents: pd.DataFrame, sentence_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-source descriptive statistics; no assumptions of normality."""
    rows = []
    for source, frame in documents.groupby("source", sort=False):
        sentence_words = sentence_rows.loc[
            sentence_rows["source"].eq(source), "n_words"
        ].astype(float)
        entry = {"source": source, "n_documents": len(frame), "n_sentences_total": len(sentence_words)}
        for key, values in (
            ("sentences_per_document", frame["n_sentences"].astype(float)),
            ("words_per_document", frame["n_words"].astype(float)),
            ("words_per_sentence", sentence_words),
        ):
            entry[f"{key}_mean"] = float(values.mean())
            entry[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            entry[f"{key}_median"] = float(values.median())
            entry[f"{key}_q05"] = float(values.quantile(0.05))
            entry[f"{key}_q25"] = float(values.quantile(0.25))
            entry[f"{key}_q75"] = float(values.quantile(0.75))
            entry[f"{key}_q95"] = float(values.quantile(0.95))
        rows.append(entry)
    return pd.DataFrame(rows)


def _save_figures(documents: pd.DataFrame, sentence_rows: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric, frame, column, title, xlabel in (
        ("words_per_sentence", sentence_rows, "n_words", "Words per sentence", "Words"),
        ("sentences_per_document", documents, "n_sentences", "Sentences per document", "Sentences"),
        ("words_per_document", documents, "n_words", "Words per document", "Words"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        # ECDF works for discrete values and strongly skewed distributions.
        sns.ecdfplot(data=frame, x=column, hue="source", ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel="Cumulative proportion")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{metric}_ecdf.png", dpi=220)
        plt.close(fig)


def select_matched_dataset(
    documents: pd.DataFrame,
    sentences: pd.DataFrame,
    *,
    k: int,
    m: int | None = None,
    domain: str | None = None,
    prompting=None,
    attack: str | None = "null",
    decoding: str | None = "greedy",
    repetition_penalty: str | None = "no",
    input_file: str | Path | None = None,
    output_root: str | Path | None = None,
    seed: int = RANDOM_SEED,
    chunksize: int = CHUNK_SIZE,
    save: bool = True,
    plot: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (matched_documents, summary_statistics, metadata).

    Each source_id is kept iff it has exactly one eligible document for *every*
    source in MODEL_ORDER (including Human). The k condition applies to full
    documents; no truncation is performed. ``m`` counts groups and thus also
    documents *per source*.

    Analyzed input must correspond to the same RAID filters. Pass the original
    full CSV via ``input_file`` if not stored in the analysis metadata.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if m is not None and (not isinstance(m, int) or isinstance(m, bool) or m < 1):
        raise ValueError("m must be None or a positive integer")
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be positive")
    _validate_statistics(documents, sentences)

    source_names = list(dict.fromkeys(MODEL_ORDER))
    if "Human" not in source_names:
        raise ValueError("MODEL_ORDER must contain Human for strict all-source matching")
    if not source_names:
        raise ValueError("MODEL_ORDER is empty")

    original_meta = dict(documents.attrs.get("raid_metadata", {}))
    if domain is None:
        domain = original_meta.get("domain")
    if domain is None:
        raise ValueError("Provide domain, or documents.attrs['raid_metadata']['domain']")
    filters = {
        "domain": _canonical_domain(domain),
        "prompting": prompting,
        "attack": attack,
        "decoding": decoding,
        "repetition_penalty": repetition_penalty,
    }
    # A cached sentence analysis cannot be repurposed for a different filter.
    for key, requested in filters.items():
        if key in original_meta and original_meta[key] != requested:
            raise ValueError(
                f"Filter mismatch for {key}: analyzed={original_meta[key]!r}, "
                f"requested={requested!r}. Re-run sentence analysis with matching filters."
            )
    if not original_meta:
        raise ValueError(
            "Missing raid_metadata on documents. Restore metadata.json to "
            "documents.attrs['raid_metadata'] before matching."
        )

    frame = documents.copy()
    frame["id"] = frame["id"].astype("string")
    frame["source_id"] = frame["source_id"].astype("string").str.strip()
    frame["source"] = frame["source"].astype("string")
    frame = frame.loc[frame["source"].isin(source_names)].copy()
    if set(source_names) - set(frame["source"].dropna()):
        raise ValueError(f"Missing required sources: {sorted(set(source_names) - set(frame['source'].dropna()))}")
    frame = frame.loc[frame["source_id"].notna() & frame["source_id"].ne("")].copy()
    frame["n_sentences"] = pd.to_numeric(frame["n_sentences"], errors="raise")
    frame["n_words"] = pd.to_numeric(frame["n_words"], errors="raise")
    # Duplicate groups are ambiguous even if only one duplicate satisfies k.
    duplicates = frame.duplicated(["source_id", "source"], keep=False)
    duplicate_groups = set(frame.loc[duplicates, "source_id"].tolist())
    if duplicate_groups:
        frame = frame.loc[~frame["source_id"].isin(duplicate_groups)].copy()

    eligible = frame.loc[frame["n_sentences"].ge(k)].copy()
    group_counts = eligible.groupby("source_id")["source"].nunique()
    matched_ids = group_counts.index[group_counts.eq(len(source_names))].to_numpy(dtype=str)
    matched_ids.sort()
    available = len(matched_ids)
    if not available:
        raise ValueError(f"No complete source_id groups with at least k={k} sentences per source")
    if m is not None and m > available:
        raise ValueError(f"Requested m={m} groups but only {available} matched groups are available")
    if m is not None:
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(matched_ids, size=m, replace=False).tolist())
    else:
        chosen = set(matched_ids.tolist())
    selected = eligible.loc[eligible["source_id"].isin(chosen)].copy()
    selected = selected.sort_values(["source_id", "source", "id"]).reset_index(drop=True)
    if len(selected) != len(chosen) * len(source_names):
        raise AssertionError("Matching invariant violated: incomplete selected groups")

    source_csv = Path(input_file or original_meta.get("input_file") or locate_training_file())
    ids = set(selected["id"].astype(str))
    source_by_id = dict(zip(selected["id"].astype(str), selected["source"].astype(str)))
    full_texts = _recover_full_texts(ids, source_by_id, source_csv, filters, chunksize=chunksize)
    selected["text"] = selected["id"].astype(str).map(full_texts)
    selected["words_per_sentence_mean"] = selected["n_words"] / selected["n_sentences"]

    sentence_frame = sentences.copy()
    sentence_frame["id"] = sentence_frame["id"].astype("string")
    sentence_frame = sentence_frame.loc[sentence_frame["id"].isin(ids)].copy()
    sentence_frame = sentence_frame.merge(
        selected[["id", "source_id", "source"]], on="id", how="left", validate="many_to_one"
    )
    sentence_frame["n_words"] = pd.to_numeric(sentence_frame["n_words"], errors="raise")
    observed = sentence_frame.groupby("id").size().reindex(selected["id"], fill_value=0).to_numpy()
    if not np.array_equal(observed, selected["n_sentences"].to_numpy(dtype=int)):
        raise ValueError("Sentence records are incomplete or inconsistent with document statistics")
    summary = _source_summary(selected, sentence_frame)

    meta = {
        **filters,
        "selection_version": SELECTION_VERSION,
        "k": k,
        "m": m,
        "seed": seed if m is not None else None,
        "match_source_id": True,
        "text_policy": "original complete RAID text, never truncated",
        "sources": source_names,
        "n_sources": len(source_names),
        "n_matched_groups_available": available,
        "n_matched_groups_selected": len(chosen),
        "n_documents_selected": len(selected),
        "n_source_id_groups_discarded_due_to_duplicates": len(duplicate_groups),
        "source_csv": str(source_csv.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    selected.attrs["raid_metadata"] = meta
    summary.attrs["raid_metadata"] = meta

    if save:
        folder = _output_directory(_resolve_filtered_root(output_root), filters, k, m, seed)
        folder.mkdir(parents=True, exist_ok=True)
        selected.to_csv(folder / "matched_documents.csv", index=False)
        sentence_frame.to_csv(folder / "sentence_statistics.csv", index=False)
        summary.to_csv(folder / "statistics.csv", index=False)
        with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        if plot:
            _save_figures(selected, sentence_frame, folder / "figures")
        print(f"Saved matched dataset: {folder}")
    print(f"Matched groups: {len(chosen):,} / {available:,} available; "
          f"{len(selected):,} documents across {len(source_names)} sources")
    return selected, summary, meta


__all__ = ["select_matched_dataset"]
