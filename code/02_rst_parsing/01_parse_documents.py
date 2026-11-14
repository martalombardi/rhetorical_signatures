"""RST parsing of cached/selected RAID documents.

Loads the same preprocessing selection as 02_process_sentences.py; outputs
resumable JSONL in data/rst/{matched,unmatched}/<experiment>/<selection>/.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
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



def parse_dataset(
    *,
    domain,
    n_sentences,
    prompting=None,
    attack="null",
    decoding="greedy",
    repetition_penalty="no",
    matched=False,
    sampling="proportional",
    seed=None,
    n_bins=None,
    include_human=True,
    device="auto",
    retry_errors=False,
    limit=None,
    output_root=None,
    parser_factory=None,
    min_sentences=None,
    max_sentences=None,
):
    """
    Parse an existing, previously selected RAID dataset.

    Never creates or resamples a dataset.
    Raises FileNotFoundError("dataset non creato") if the
    requested selection does not exist.
    """

    prep = _load_preprocessing()

    n = int(n_sentences)
    if n < 1 or n != n_sentences:
        raise ValueError("n_sentences must be a positive integer")

    if seed is None:
        seed = prep.RANDOM_SEED

    if n_bins is None:
        n_bins = prep.LENGTH_BINS

    # --------------------------------------------------------
    # Locate the existing experiment
    # --------------------------------------------------------

    analysis_meta = {
        "domain": prep._domain_key(domain),
        "prompting": prompting,
        "attack": attack,
        "decoding": decoding,
        "repetition_penalty": repetition_penalty,
    }

    run = prep._run_name(analysis_meta)

    selection_root = (
        ROOT / "data" / "filtered" / "sentences" / run
    )

    # --------------------------------------------------------
    # Find an existing selection for exactly n sentences
    # --------------------------------------------------------

    selection_suffix = (
        f"__best-{n}"
        f"__matched-{int(matched)}"
        f"__{sampling}"
        f"__seed-{seed}"
        f"__bins-{n_bins}"
        f"__human-{int(include_human)}"
    )

    if not selection_root.is_dir():
        raise FileNotFoundError(
            f"dataset non creato: {selection_root}"
        )

    candidates = sorted(
        path
        for path in selection_root.iterdir()
        if path.is_dir()
        and path.name.startswith("range-")
        and path.name.endswith(selection_suffix)
        and (path / "selected_documents.csv").is_file()
    )

    # Optionally restrict the search to the exact
    # preprocessing range used in the notebook.
    if min_sentences is not None and max_sentences is not None:
        expected_prefix = (
            f"range-{min_sentences}-{max_sentences}__"
        )
        candidates = [
            path for path in candidates
            if path.name.startswith(expected_prefix)
        ]

    if not candidates:
        raise FileNotFoundError(
            "dataset non creato: "
            f"no saved selection for n_sentences={n}, "
            f"matched={matched}, run={run}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple matching datasets found. "
            "Specify min_sentences and max_sentences "
            "to identify the intended selection:\n"
            + "\n".join(str(p) for p in candidates)
        )

    selection_dir = candidates[0]

    # --------------------------------------------------------
    # Load selected documents and cached sentences
    # --------------------------------------------------------

    selected = pd.read_csv(
        selection_dir / "selected_documents.csv"
    )

    sentences_file = (
        ROOT
        / "results"
        / "sentences_report"
        / run
        / "sentences.csv"
    )

    if not sentences_file.is_file():
        raise FileNotFoundError(
            f"Sentence analysis not found: {sentences_file}"
        )

    sentences = pd.read_csv(sentences_file)

    required_selected = {
        "id", "source_id", "source"
    }
    required_sentences = {
        "id", "sentence_idx", "sentence_text"
    }

    if not required_selected.issubset(selected.columns):
        raise ValueError(
            "Missing columns in selected_documents.csv: "
            f"{required_selected - set(selected.columns)}"
        )

    if not required_sentences.issubset(sentences.columns):
        raise ValueError(
            "Missing columns in sentences.csv: "
            f"{required_sentences - set(sentences.columns)}"
        )

    if selected["id"].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate document IDs in selected dataset"
        )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    base = (
        Path(output_root)
        if output_root is not None
        else ROOT / "data" / "rst"
    )

    out = (
        base
        / ("matched" if matched else "unmatched")
        / run
        / selection_dir.name
    )

    out.mkdir(parents=True, exist_ok=True)

    trees_file = out / "trees.jsonl"
    errors_file = out / "errors.jsonl"

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    done = _read_ids(trees_file)

    errors = (
        set()
        if retry_errors
        else _read_ids(errors_file)
    )

    eligible = selected.loc[
        ~selected["id"].astype(str).isin(done | errors)
    ].copy()

    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        eligible = eligible.head(limit)

    # --------------------------------------------------------
    # Recover exactly the first n sentences
    # --------------------------------------------------------

    wanted = set(eligible["id"].astype(str))

    first = sentences.loc[
        sentences["id"].astype(str).isin(wanted)
        & sentences["sentence_idx"].between(1, n),
        ["id", "sentence_idx", "sentence_text"],
    ].copy()

    first = first.sort_values(
        ["id", "sentence_idx"]
    )

    grouped = {
        str(key): group
        for key, group in first.groupby("id", sort=False)
    }

    for row in eligible.itertuples(index=False):
        group = grouped.get(str(row.id))

        if (
            group is None
            or len(group) != n
            or group["sentence_idx"].tolist()
               != list(range(1, n + 1))
        ):
            raise ValueError(
                f"Missing or duplicated first {n} sentences "
                f"for document {row.id}"
            )

    # --------------------------------------------------------
    # Initialize parser
    # --------------------------------------------------------

    parser = None

    if not eligible.empty:
        if parser_factory is None:
            import torch
            from isanlp_rst.parser import Parser

            if device not in ("auto", "cpu", "cuda"):
                raise ValueError(
                    "device must be 'auto', 'cpu', or 'cuda'"
                )

            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA requested but not available"
                )

            use_cuda = (
                torch.cuda.is_available()
                if device == "auto"
                else device == "cuda"
            )

            parser = Parser(
                hf_model_name="tchewik/isanlp_rst_v3",
                hf_model_version="rstdt",
                cuda_device=0 if use_cuda else -1,
            )
        else:
            parser = parser_factory()

    # --------------------------------------------------------
    # Parse documents
    # --------------------------------------------------------

    for row in tqdm(
        eligible.itertuples(index=False),
        total=len(eligible),
        desc=f"RST parsing (n={n})",
    ):
        text = "\n".join(
            grouped[str(row.id)]["sentence_text"].astype(str)
        )

        base_record = {
            "id": str(row.id),
            "source_id": str(row.source_id),
            "source": str(row.source),
            "n_sentences": n,
            "text": text,
        }

        try:
            result = parser(text)
            tree = result["rst"][0]

            _append(
                trees_file,
                {
                    **base_record,
                    "rst_raw_type": (
                        f"{type(tree).__module__}."
                        f"{type(tree).__qualname__}"
                    ),
                    "rst_result": _jsonable(result),
                    "parsed_utc": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                },
            )

        except Exception as exc:
            _append(
                errors_file,
                {
                    **base_record,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_utc": (
                        datetime.now(timezone.utc).isoformat()
                    ),
                },
            )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {
        "dataset_path": str(selection_dir),
        "run": run,
        "n_sentences": n,
        "matched": matched,
        "sampling": sampling,
        "seed": seed,
        "n_bins": n_bins,
        "include_human": include_human,
        "parser_model": "tchewik/isanlp_rst_v3",
        "parser_version": "rstdt",
        "total_selected": len(selected),
        "successful_ids": len(_read_ids(trees_file)),
        "error_ids": len(_read_ids(errors_file)),
        "output_directory": str(out),
        "updated_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
    }

    (out / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Loaded existing dataset: {selection_dir}")
    print(f"RST output: {out}")

    return out