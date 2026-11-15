"""RST parsing of cached/selected RAID documents.

Loads the same preprocessing selection as 02_process_sentences.py; outputs
resumable JSONL in data/rst/{matched,unmatched}/<experiment>/<selection>/.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import hashlib
import math
from pathlib import Path
import sys
import traceback

import pandas as pd
from tqdm.auto import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
ROOT = CODE_DIR.parent
PREPROCESS = CODE_DIR / '01_data_processing' / '02_process_sentences.py'


def _load_preprocessing():
    spec = importlib.util.spec_from_file_location('raid_sentence_processing', PREPROCESS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {PREPROCESS}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value, seen=None):
    """Preserve parser output, including custom tree objects, as JSON data."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, seen) for v in value]
    if hasattr(value, 'item'):
        try:
            return _jsonable(value.item(), seen)
        except (ValueError, TypeError, AttributeError):
            pass
    if hasattr(value, 'tolist'):
        try:
            return _jsonable(value.tolist(), seen)
        except (ValueError, TypeError, AttributeError):
            pass
    oid = id(value)
    if oid in seen:
        return {'__reference__': type(value).__name__}
    seen.add(oid)
    try:
        if hasattr(value, 'to_dict') and callable(value.to_dict):
            try:
                return _jsonable(value.to_dict(), seen)
            except (TypeError, ValueError, AttributeError):
                pass
        if hasattr(value, '__dict__'):
            return {'__class__': f'{type(value).__module__}.{type(value).__qualname__}',
                    '__attributes__': {k: _jsonable(v, seen) for k, v in vars(value).items()}}
        if hasattr(value, '__slots__'):
            return {'__class__': f'{type(value).__module__}.{type(value).__qualname__}',
                    '__attributes__': {k: _jsonable(getattr(value, k), seen)
                                       for k in value.__slots__ if hasattr(value, k)}}
        raise TypeError(f'Unsupported parser output type: {type(value)!r}')
    finally:
        seen.remove(oid)


def _read_ids(path):
    if not path.is_file():
        return set()
    ids = set()
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(str(json.loads(line)['id']))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f'Invalid JSONL in {path}: {exc}') from exc
    return ids


def _append(path, item):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()



def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path):
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or "id" not in record:
                    raise ValueError("missing document ID")
                records.append(record)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid JSONL: {path}, line {number}: {exc}") from exc
    return records


def _atomic_manifest(path, manifest):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def parse_dataset(
    *, domain, n_sentences, prompting=None, attack="null", decoding="greedy",
    repetition_penalty="no", matched=False, sampling="proportional",
    seed=None, n_bins=None, include_human=True, device="auto",
    retry_errors=False, limit=None, output_root=None, parser_factory=None,
    min_sentences=None, max_sentences=None,
    model="tchewik/isanlp_rst_v3", model_version="rstdt", rebuild=False,
    _max_per_source=None,
):
    """Parse a saved selection, resuming successful documents and recording failures.

    Never creates or resamples the selection. A cache with incompatible metadata
    raises an error unless rebuild=True, to prevent mixing parser configurations.
    """
    prep = _load_preprocessing()
    n = int(n_sentences)
    if n < 1 or n != n_sentences:
        raise ValueError("n_sentences must be a positive integer")
    if seed is None:
        seed = prep.RANDOM_SEED
    if n_bins is None:
        n_bins = prep.LENGTH_BINS
    if limit is not None and (not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be None or a non-negative integer")
    if _max_per_source is not None and (not isinstance(_max_per_source, int) or _max_per_source < 1):
        raise ValueError("_max_per_source must be a positive integer")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")

    analysis_meta = dict(
        domain=prep._domain_key(domain), prompting=prompting, attack=attack,
        decoding=decoding, repetition_penalty=repetition_penalty,
    )
    run = prep._run_name(analysis_meta)
    selection_root = ROOT / "data" / "filtered" / "sentences" / run
    suffix = (
        f"__best-{n}__matched-{int(matched)}__{sampling}"
        f"__seed-{seed}__bins-{n_bins}__human-{int(include_human)}"
    )
    if not selection_root.is_dir():
        raise FileNotFoundError(f"dataset non creato: {selection_root}")
    candidates = sorted(
        path for path in selection_root.iterdir()
        if path.is_dir() and path.name.startswith("range-")
        and path.name.endswith(suffix)
        and (path / "selected_documents.csv").is_file()
    )
    if min_sentences is not None and max_sentences is not None:
        prefix = f"range-{min_sentences}-{max_sentences}__"
        candidates = [p for p in candidates if p.name.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(
            f"dataset non creato: no saved selection for n_sentences={n}, "
            f"matched={matched}, run={run}. Check N_SENTENCES against "
            "the selection's best_n_sentences."
        )
    if len(candidates) != 1:
        raise RuntimeError("Multiple matching selections:\n" + "\n".join(map(str, candidates)))

    selection_dir = candidates[0]
    selected_file = selection_dir / "selected_documents.csv"
    selected = pd.read_csv(selected_file, dtype={"id": "string", "source_id": "string"})
    required = {"id", "source_id", "source"}
    if not required.issubset(selected.columns):
        raise ValueError(f"Missing selected columns: {required - set(selected.columns)}")
    if selected["id"].isna().any() or selected["id"].duplicated().any():
        raise ValueError("Selected documents have null or duplicate IDs")
    expected_ids = set(selected["id"].astype(str))

    base = Path(output_root) if output_root is not None else ROOT / "data" / "rst"
    out = base / ("matched" if matched else "unmatched") / run / selection_dir.name
    out.mkdir(parents=True, exist_ok=True)
    trees_file, errors_file = out / "trees.jsonl", out / "errors.jsonl"
    manifest_file = out / "manifest.json"

    identity = dict(
        dataset_path=str(selection_dir), selection_sha256=_fingerprint(selected_file),
        run=run, n_sentences=n, matched=bool(matched), sampling=sampling,
        seed=seed, n_bins=n_bins, include_human=bool(include_human),
        parser_model=model, parser_version=model_version,
    )
    if rebuild:
        for path in (trees_file, errors_file, manifest_file):
            path.unlink(missing_ok=True)
    elif trees_file.exists() or errors_file.exists():
        if not manifest_file.is_file():
            raise RuntimeError(
                f"RST cache exists without manifest: {out}. "
                "Cannot verify parser settings; set RST_REBUILD=True to regenerate."
            )
        with manifest_file.open(encoding="utf-8") as stream:
            old = json.load(stream)
        # Older manifests do not include selection_sha256; compare available
        # fields and allow the legacy cache when its other settings match.
        mismatches = [
            key for key, value in identity.items()
            if key in old and old[key] != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Incompatible cached RST output at {out}; changed: {mismatches}. "
                "Set RST_REBUILD=True to regenerate."
            )

    done = _read_ids(trees_file)
    failed = _read_ids(errors_file)
    if not done.issubset(expected_ids) or not failed.issubset(expected_ids):
        raise RuntimeError("Cached RST IDs do not match the selected dataset. Rebuild required.")
    # Successful records always take precedence over older failure records.
    pending = selected.loc[
        ~selected["id"].astype(str).isin(done | (set() if retry_errors else failed))
    ].copy()
    if _max_per_source is not None:
        from collections import Counter

        # Preserve existing successful trees; never delete or reparse them.
        existing_counts = Counter()
        for record in _read_jsonl(trees_file):
            existing_counts[str(record["source"])] += 1

        # Select at most the missing quota per source, in dataset order.
        quotas = {
            str(src): max(0, _max_per_source - existing_counts[str(src)])
            for src in selected["source"].dropna().unique()
        }
        pending = pending.loc[
            pending.groupby("source", sort=False).cumcount()
            < pending["source"].astype(str).map(quotas).fillna(0)
        ].copy()
    if limit is not None:
        pending = pending.head(limit)

    print(f"RST cache: {len(done)}/{len(expected_ids)} successful; "
          f"{len(failed - done)} previous failures; {len(pending)} to parse")

    grouped = {}
    if not pending.empty:
        sentences_file = ROOT / "results" / "sentences_report" / run / "sentences.csv"
        if not sentences_file.is_file():
            raise FileNotFoundError(f"Sentence analysis not found: {sentences_file}")
        sentences = pd.read_csv(sentences_file, dtype={"id": "string"})
        columns = {"id", "sentence_idx", "sentence_text"}
        if not columns.issubset(sentences.columns):
            raise ValueError(f"Missing sentence columns: {columns - set(sentences.columns)}")
        wanted = set(pending["id"].astype(str))
        first = sentences.loc[
            sentences["id"].astype(str).isin(wanted)
            & sentences["sentence_idx"].between(1, n),
            ["id", "sentence_idx", "sentence_text"],
        ].sort_values(["id", "sentence_idx"])
        grouped = {str(key): group for key, group in first.groupby("id", sort=False)}
        for row in pending.itertuples(index=False):
            group = grouped.get(str(row.id))
            if (group is None or len(group) != n
                    or group["sentence_idx"].tolist() != list(range(1, n + 1))
                    or group["sentence_text"].isna().any()):
                raise ValueError(f"Missing or duplicated first {n} sentences for {row.id}")

    # Write a manifest BEFORE parsing, so an interrupted run remains resumable.
    def update_manifest():
        success = _read_ids(trees_file)
        errors = _read_ids(errors_file) - success
        _atomic_manifest(manifest_file, {
            **identity, "total_selected": len(selected),
            "successful_ids": len(success), "error_ids": len(errors),
            "output_directory": str(out),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })

    update_manifest()
    parser = None
    if not pending.empty:
        if parser_factory is None:
            import torch
            from isanlp_rst.parser import Parser
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            use_cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
            parser = Parser(
                hf_model_name=model, hf_model_version=model_version,
                cuda_device=0 if use_cuda else -1,
            )
        else:
            parser = parser_factory()

    for row in tqdm(pending.itertuples(index=False), total=len(pending),
                    desc=f"RST parsing (n={n})"):
        text = "\n".join(grouped[str(row.id)]["sentence_text"].astype(str))
        record = dict(
            id=str(row.id), source_id=str(row.source_id), source=str(row.source),
            n_sentences=n, text=text,
        )
        try:
            result = parser(text)
            tree = result["rst"][0]
            _append(trees_file, {
                **record,
                "rst_raw_type": f"{type(tree).__module__}.{type(tree).__qualname__}",
                "rst_result": _jsonable(result),
                "parsed_utc": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            _append(errors_file, {
                **record, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "failed_utc": datetime.now(timezone.utc).isoformat(),
            })
        update_manifest()

    print(f"Loaded existing dataset: {selection_dir}")
    print(f"RST output: {out}")
    return out

def load_or_parse_dataset(**kwargs):
    """Load cached RST trees, parsing missing documents if necessary."""

    output = parse_dataset(**kwargs)

    trees_file = output / "trees.jsonl"

    if not trees_file.is_file():
        raise FileNotFoundError(
            f"RST trees not found: {trees_file}"
        )

    with trees_file.open(encoding="utf-8") as f:
        trees = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    return output, trees

def load_or_parse_first_per_source(*, max_per_source=200, **kwargs):
    """Reuse cached trees, or parse only up to max_per_source per source.

    Writes to the EXACT SAME canonical output directory as parse_dataset().
    Does not truncate or overwrite an existing full trees.jsonl.
    Returns (output_directory, limited_trees, limited_jsonl_path).
    """
    from collections import Counter

    if not isinstance(max_per_source, int) or max_per_source < 1:
        raise ValueError("max_per_source must be a positive integer")
    if kwargs.get("rebuild", False):
        raise ValueError("rebuild=True would delete existing trees; use rebuild=False")
    if kwargs.get("limit") is not None:
        raise ValueError("Do not set limit; max_per_source controls the per-source quota")

    output = parse_dataset(
        **kwargs, rebuild=False, _max_per_source=max_per_source
    )

    trees = []
    counts = Counter()
    for record in _read_jsonl(output / "trees.jsonl"):
        source = str(record["source"])
        if counts[source] < max_per_source:
            trees.append(record)
            counts[source] += 1

    limited_path = output / f"trees_first{max_per_source}_per_source.jsonl"
    temporary = limited_path.with_name(limited_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in trees:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(limited_path)

    print("RST trees selected per source:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"Total selected: {len(trees)}")
    print(f"Full cache: {output / 'trees.jsonl'}")
    print(f"Forest input: {limited_path}")
    return output, trees, limited_path
