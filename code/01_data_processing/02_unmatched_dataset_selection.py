"""RAID unmatched selection: exactly k initial sentences per document.

No source_id matching is performed. Each configured source contributes the
same number of documents: m when specified, otherwise the minimum eligible
source count. Stratified sampling approximates each source's own first-k
word-length distribution; samples are drawn without replacement.

The shared sentence-analysis cache remains compatible with the original RAID
preprocessing module. Results go to data/filtered/unmatched/.
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
    CHUNK_SIZE, BATCH_SIZE, RANDOM_SEED, LENGTH_BINS,
    SPACY_MODEL, DOMAIN_NAMES, MODEL_NAMES, MODEL_ORDER,
    RESULTS_DIR, locate_training_file,
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


def _output_directory(root: Path, filters: dict, k: int, m: int | None, seed: int, n_bins: int) -> Path:
    parts = [
        "unmatched",
        f"genre-{_slug(filters['domain'])}",
        f"prompting-{_slug(filters['prompting'])}",
        f"attack-{_slug(filters['attack'])}",
        f"decoding-{_slug(filters['decoding'])}",
        f"repetition_penalty-{_slug(filters['repetition_penalty'])}",
        f"k-{k}",
        f"balanced-all__bins-{n_bins}__seed-{seed}" if m is None else f"m-{m}__bins-{n_bins}__seed-{seed}",
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
    if sentences[["id", "sentence_idx"]].isna().any().any() or sentences.duplicated(["id", "sentence_idx"]).any():
        raise ValueError("Sentence IDs and indices must be non-null and unique")
    if documents[["n_sentences", "n_words"]].isna().any().any():
        raise ValueError("Missing document statistics")


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


def _domain_key(domain):
    return _canonical_domain(domain)

def _match(series, value):
    if value is None:
        return pd.Series(True, index=series.index)
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError('Empty filter collection')
    allowed = {str(x).strip().lower() for x in values}
    return series.astype('string').str.strip().str.lower().isin(allowed).fillna(False)

def _run_name(meta):
    return '__'.join([
        f"domain-{_slug(meta['domain'])}",
        f"prompting-{_slug(meta['prompting'])}",
        f"attack-{_slug(meta['attack'])}",
        f"decoding-{_slug(meta['decoding'])}",
        f"rep-{_slug(meta['repetition_penalty'])}",
    ])

def _filter_chunk(chunk, meta):
    cols = {
        'domain': _find(chunk.columns, 'domain', 'genre'),
        'model': _find(chunk.columns, 'model', 'source'),
        'text': _find(chunk.columns, 'generation', 'text', 'response'),
        'id': _find(chunk.columns, 'id'),
        'source_id': _find(chunk.columns, 'source_id'),
    }
    aliases = {
        'attack': ('attack', 'attack_type', 'attack_name'),
        'decoding': ('decoding', 'decoding_strategy', 'decoding_method'),
        'repetition_penalty': ('repetition_penalty', 'repetition_penalty_enabled'),
        'prompting': ('prompting', 'prompting_strategy', 'prompting_method', 'prompt_type', 'prompt'),
    }
    for key, names in aliases.items():
        if meta[key] is not None:
            cols[key] = _find(chunk.columns, *names)
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError(f'Missing RAID columns: {missing}; available: {list(chunk.columns)}')

    models = chunk[cols['model']].astype('string').str.strip().str.lower()
    human = models.eq('human').fillna(False)
    mask = (_match(chunk[cols['domain']], meta['domain']) & models.isin(MODEL_NAMES))
    attack = meta['attack']
    if attack is not None:
        if isinstance(attack, str) and attack.strip().lower() == 'null':
            a = chunk[cols['attack']].astype('string').str.strip().str.lower()
            mask &= (a.isna() | a.isin(['', 'none', 'null'])).fillna(False)
        else:
            mask &= _match(chunk[cols['attack']], attack)
    for key in ('decoding', 'repetition_penalty', 'prompting'):
        if meta[key] is not None:
            mask &= (human | _match(chunk[cols[key]], meta[key]))
    text = chunk[cols['text']]
    mask &= text.notna() & text.astype('string').str.strip().str.len().gt(0).fillna(False)
    selected = chunk.loc[mask]
    return pd.DataFrame({
        'id': selected[cols['id']].to_numpy(),
        'source_id': selected[cols['source_id']].to_numpy(),
        'source': models.loc[mask].map(MODEL_NAMES).to_numpy(),
        'text': selected[cols['text']].to_numpy(),
    })

def _save_metadata(folder, meta):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

def analyze_sentence_lengths(
    domain, prompting=None, attack='null', decoding='greedy',
    repetition_penalty='no', *, input_file=None, batch_size=BATCH_SIZE,
    chunksize=CHUNK_SIZE, save=True, show_progress=True,
):
    """Return (documents, sentences) for all filtered RAID records.

    documents: id, source_id, source, n_sentences, n_words.
    sentences: id, sentence_idx (1-based), n_words, sentence_text.
    Full document text is not duplicated in these outputs. Each sentence is
    retained once so the first n sentences can later be reconstructed for RST.
    """
    import spacy
    from tqdm.auto import tqdm

    meta = dict(domain=_domain_key(domain), prompting=prompting, attack=attack,
                decoding=decoding, repetition_penalty=repetition_penalty)
    path = Path(input_file) if input_file is not None else locate_training_file()
    if not path.is_file():
        raise FileNotFoundError(path)
    nlp = spacy.load(SPACY_MODEL, disable=['ner', 'tagger', 'lemmatizer'])
    nlp.max_length = 2_000_000
    document_rows, sentence_rows = [], []
    progress = tqdm(desc='RAID sentence parsing', unit='docs', disable=not show_progress)
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype='string'):
            selected = _filter_chunk(chunk, meta)
            if selected.empty:
                continue
            if selected['id'].isna().any():
                raise ValueError('Null document IDs found')
            texts = selected['text'].astype(str).tolist()
            for record, parsed in zip(selected.itertuples(index=False),
                                      nlp.pipe(texts, batch_size=batch_size, n_process=1)):
                spans = list(parsed.sents)
                # Use whitespace-separated word counts, as in the original notebook.
                n_words = len(record.text.split())
                document_rows.append((record.id, record.source_id, record.source,
                                      len(spans), n_words))
                for j, span in enumerate(spans, 1):
                    sentence_rows.append((record.id, j, len(span.text.split()), span.text))
                progress.update(1)
    finally:
        progress.close()
    documents = pd.DataFrame(document_rows,
                             columns=['id', 'source_id', 'source', 'n_sentences', 'n_words'])
    sentences = pd.DataFrame(sentence_rows,
                             columns=['id', 'sentence_idx', 'n_words', 'sentence_text'])
    if documents.empty:
        raise ValueError('No documents match the requested RAID filters')
    if documents['id'].duplicated().any():
        raise ValueError('Duplicate document IDs: cannot safely join sentence rows')
    meta.update(input_file=str(path.resolve()), n_documents=len(documents),
                n_sentences_total=len(sentences),
                created_utc=datetime.now(timezone.utc).isoformat(),
                segmentation='spacy en_core_web_sm dependency parser',
                word_count='Python whitespace split')
    documents.attrs['raid_metadata'] = meta
    sentences.attrs['raid_metadata'] = meta
    if save:
        folder = RESULTS_DIR / _run_name(meta)
        _save_metadata(folder, meta)
        documents.to_csv(folder / 'documents.csv', index=False)
        sentences.to_csv(folder / 'sentences.csv', index=False)
        print('Saved sentence analysis:', folder)
    return documents, sentences

def load_or_analyze_sentence_lengths(
    domain,
    prompting=None,
    attack="null",
    decoding="greedy",
    repetition_penalty="no",
    *,
    rebuild=False,
    show_progress=True,
):
    """Load cached sentence analysis or compute it if necessary.

    Cached results are reused only when the requested RAID
    filters match the saved metadata.

    Set rebuild=True to force recomputation.
    """

    filters = dict(
        domain=_domain_key(domain),
        prompting=prompting,
        attack=attack,
        decoding=decoding,
        repetition_penalty=repetition_penalty,
    )

    folder = RESULTS_DIR / _run_name(filters)

    documents_path = folder / "documents.csv"
    sentences_path = folder / "sentences.csv"
    metadata_path = folder / "metadata.json"

    cached_files_exist = all(
        path.is_file()
        for path in (
            documents_path,
            sentences_path,
            metadata_path,
        )
    )

    if cached_files_exist and not rebuild:

        with metadata_path.open(
            "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        metadata_matches = all(
            metadata.get(key) == value
            for key, value in filters.items()
        )

        if metadata_matches:

            documents = pd.read_csv(
                documents_path,
                dtype={
                    "id": "string",
                    "source_id": "string",
                },
            )

            sentences = pd.read_csv(
                sentences_path,
                dtype={"id": "string"},
            )

            documents.attrs["raid_metadata"] = metadata
            sentences.attrs["raid_metadata"] = metadata

            print(
                f"Loaded existing sentence analysis: {folder}"
            )

            return documents, sentences

        print(
            "Cached metadata differs from the requested "
            "experiment. Recomputing sentence analysis."
        )

    elif rebuild:
        print(
            "Rebuild requested: recomputing sentence analysis."
        )

    else:
        print(
            "No cached sentence analysis found. "
            "Computing results."
        )

    return analyze_sentence_lengths(
        domain=domain,
        prompting=prompting,
        attack=attack,
        decoding=decoding,
        repetition_penalty=repetition_penalty,
        save=True,
        show_progress=show_progress,
    )

def _proportional_sample(group: pd.DataFrame, target: int, rng: np.random.Generator,
                         n_bins: int) -> pd.DataFrame:
    """Largest-remainder allocation across source-specific quantile strata."""
    if target == len(group):
        return group.copy()
    if not 0 < target < len(group):
        raise ValueError("Invalid target for source-specific sampling")
    # Stable sorting guarantees reproducible sampling regardless of input row order.
    group = group.sort_values("id", kind="stable")
    # Rank breaks ties in discrete word counts; strata have approximately equal sizes.
    ranks = group["first_k_words"].rank(method="first")
    labels = pd.qcut(ranks, q=min(n_bins, len(group)), labels=False).astype(int)
    counts = labels.value_counts().sort_index()
    quotas = counts.to_numpy(dtype=float) * target / len(group)
    allocations = np.floor(quotas).astype(int)
    remaining = target - int(allocations.sum())
    if remaining:
        # Stable tie-breaking by stratum index.
        order = np.argsort(-(quotas - allocations), kind="stable")
        allocations[order[:remaining]] += 1
    indices = []
    for label, amount in zip(counts.index, allocations):
        if amount:
            candidates = group.index[labels.eq(label)].to_numpy()
            indices.extend(rng.choice(candidates, size=int(amount), replace=False).tolist())
    return group.loc[indices].copy()


def prepare_unmatched_eligible(documents: pd.DataFrame, sentences: pd.DataFrame,
                               k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return eligible documents with their exact first-k text and first-k sentence rows."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    _validate_statistics(documents, sentences)
    frame = documents.copy()
    frame["id"] = frame["id"].astype("string")
    frame["n_sentences"] = pd.to_numeric(frame["n_sentences"], errors="raise")
    frame = frame.loc[frame["n_sentences"].ge(k)].copy()
    first = sentences.loc[
        sentences["id"].astype("string").isin(frame["id"])
        & pd.to_numeric(sentences["sentence_idx"], errors="raise").le(k),
        ["id", "sentence_idx", "n_words", "sentence_text"],
    ].copy()
    first["id"] = first["id"].astype("string")
    first["sentence_idx"] = pd.to_numeric(first["sentence_idx"], errors="raise")
    first["n_words"] = pd.to_numeric(first["n_words"], errors="raise")
    if first["sentence_text"].isna().any():
        raise ValueError("Missing sentence_text in sentence cache")
    first = first.sort_values(["id", "sentence_idx"], kind="stable")
    grouped = first.groupby("id", sort=False).agg(
        first_k_words=("n_words", "sum"),
        text=("sentence_text", " ".join),
        sentence_count=("sentence_idx", "size"),
    )
    frame = frame.join(grouped, on="id", validate="one_to_one")
    if frame["sentence_count"].isna().any() or not frame["sentence_count"].eq(k).all():
        raise ValueError("Incomplete first-k sentence rows for eligible documents")
    frame["first_k_words"] = frame["first_k_words"].astype(int)
    frame["mean_words_per_sentence"] = frame["first_k_words"] / k
    frame = frame.drop(columns="sentence_count")
    first = first.merge(frame[["id", "source"]], on="id", validate="many_to_one")
    return frame, first


def _distribution_report(eligible: pd.DataFrame, selected: pd.DataFrame,
                         sources: list[str]) -> pd.DataFrame:
    rows = []
    for source in sources:
        population = eligible.loc[eligible["source"].eq(source), "first_k_words"]
        sample = selected.loc[selected["source"].eq(source), "first_k_words"]
        # KS distance between empirical CDFs, no scipy dependency.
        values = np.sort(np.unique(np.r_[population.to_numpy(), sample.to_numpy()]))
        p = np.searchsorted(np.sort(population), values, side="right") / len(population)
        q = np.searchsorted(np.sort(sample), values, side="right") / len(sample)
        rows.append({"source": source, "eligible": len(population), "selected": len(sample),
                     "eligible_mean_first_k_words": float(population.mean()),
                     "selected_mean_first_k_words": float(sample.mean()),
                     "eligible_median_first_k_words": float(population.median()),
                     "selected_median_first_k_words": float(sample.median()),
                     "ecdf_max_difference": float(np.max(np.abs(p - q)))})
    return pd.DataFrame(rows)


def select_unmatched_dataset(
    documents: pd.DataFrame, sentences: pd.DataFrame, *, k: int,
    m: int | None = None, domain: str | None = None, prompting=None,
    attack: str | None = "null", decoding: str | None = "greedy",
    repetition_penalty: str | None = "no", output_root: str | Path | None = None,
    seed: int = RANDOM_SEED, n_bins: int = LENGTH_BINS,
    save: bool = True, plot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Balance source counts, preserving each source's eligible first-k length distribution.

    The returned text is exactly the first k spaCy sentences, not the full document.
    m=None selects the minimum eligible source count. No source_id matching.
    """
    if m is not None and (isinstance(m, bool) or not isinstance(m, int) or m < 1):
        raise ValueError("m must be None or a positive integer")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if seed is None:
        raise ValueError("Explicit integer seed required for reproducibility")
    original_meta = documents.attrs.get("raid_metadata", {})
    if domain is None:
        domain = original_meta.get("domain")
    if domain is None:
        raise ValueError("Missing domain")
    filters = dict(domain=_canonical_domain(domain), prompting=prompting,
                   attack=attack, decoding=decoding,
                   repetition_penalty=repetition_penalty)
    for key, value in filters.items():
        if key in original_meta and original_meta[key] != value:
            raise ValueError(f"Sentence-analysis filter mismatch: {key}")
    eligible, first = prepare_unmatched_eligible(documents, sentences, k)
    sources = [source for source in MODEL_ORDER if source in documents["source"].unique()]
    if not sources:
        raise ValueError("No configured sources found")
    counts = eligible["source"].value_counts().reindex(sources, fill_value=0)
    if counts.eq(0).any():
        raise ValueError(f"Sources without >=k-sentence documents: {counts[counts.eq(0)].index.tolist()}")
    target = int(counts.min()) if m is None else m
    if (counts < target).any():
        raise ValueError(f"m={target} exceeds eligible counts: {counts.to_dict()}")
    rng = np.random.default_rng(seed)
    selected = pd.concat([
        _proportional_sample(eligible.loc[eligible["source"].eq(source)], target, rng, n_bins)
        for source in sources
    ], ignore_index=True)
    selected = selected.rename(columns={"n_sentences": "original_n_sentences",
                                        "n_words": "original_n_words"})
    selected["n_sentences"] = k
    selected["n_words"] = selected["first_k_words"]
    selected = selected.sort_values(["source", "id"], kind="stable").reset_index(drop=True)
    selected_first = first.loc[first["id"].isin(selected["id"])].copy()
    if len(selected_first) != k * len(selected):
        raise AssertionError("Selected sentence count differs from k times selected documents")
    if selected["source"].value_counts().reindex(sources, fill_value=0).ne(target).any():
        raise AssertionError("Source balancing failed")
    report = _distribution_report(eligible, selected, sources)
    summary = _source_summary(selected, selected_first)
    meta = {**filters, "selection_version": 2, "selection_type": "unmatched",
            "k": k, "m": m, "effective_documents_per_source": target,
            "seed": seed, "n_bins": n_bins, "sources": sources,
            "eligible_counts": counts.astype(int).to_dict(),
            "selected_counts": {source: target for source in sources},
            "text_policy": "first k spaCy sentences only",
            "stratification_variable": "total words in first k sentences",
            "sampling": "source-specific quantile stratification, without replacement",
            "created_utc": datetime.now(timezone.utc).isoformat()}
    selected.attrs["raid_metadata"] = meta
    summary.attrs["raid_metadata"] = meta
    if save:
        folder = _output_directory(_resolve_filtered_root(output_root), filters, k, m, seed, n_bins)
        folder.mkdir(parents=True, exist_ok=True)
        selected.to_csv(folder / "unmatched_documents.csv", index=False)
        selected_first.to_csv(folder / "sentence_statistics.csv", index=False)
        summary.to_csv(folder / "statistics.csv", index=False)
        report.to_csv(folder / "distribution_preservation.csv", index=False)
        eligible[["id", "source", "first_k_words", "mean_words_per_sentence"]].to_csv(
            folder / "eligible_first_k_lengths.csv", index=False)
        with (folder / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        if plot:
            plot_unmatched_distributions(documents, sentences, selected, k, save_dir=folder / "figures")
        print(f"Saved unmatched dataset: {folder}")
    print(f"Eligible: {len(eligible):,} | balanced: {target:,} per source | selected: {len(selected):,}")
    return selected, summary, meta


def plot_unmatched_distributions(documents: pd.DataFrame, sentences: pd.DataFrame,
                                 selected: pd.DataFrame, k: int,
                                 save_dir: str | Path | None = None):
    """Plot full RAID-filtered dataset and balanced first-k unmatched dataset.

    Figure 1: all documents, without minimum-k eligibility filtering.
    Figure 2: balanced documents truncated to their first k sentences.
    The stratification itself remains based on eligible first-k word counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns

    sources = [s for s in MODEL_ORDER if s in selected["source"].unique()]
    if not sources:
        raise ValueError("No selected sources to plot")
    palette = dict(zip(sources, sns.color_palette("colorblind", len(sources))))
    handles = [Line2D([0], [0], color=palette[s], lw=2, label=s) for s in sources]

    # Original data: every document surviving the RAID filters, not just eligible ones.
    original_documents = documents.copy()
    original_sentences = sentences.merge(
        original_documents[["id", "source"]], on="id", how="inner",
        validate="many_to_one",
    )

    # Selected data: exactly the first k sentences per selected document.
    selected_ids = selected["id"].astype("string")
    selected_sentences = original_sentences.loc[
        original_sentences["id"].astype("string").isin(selected_ids)
        & pd.to_numeric(original_sentences["sentence_idx"], errors="raise").le(k)
    ].copy()
    selected_documents = selected.copy()
    if "first_k_words" not in selected_documents.columns:
        if "n_words" not in selected_documents.columns:
            raise ValueError("Selected documents need first_k_words or n_words")
        selected_documents["first_k_words"] = selected_documents["n_words"]
    selected_documents["mean_words_per_sentence"] = selected_documents["first_k_words"] / k

    counts = selected_sentences.groupby("id").size()
    if not counts.reindex(selected_ids, fill_value=0).eq(k).all():
        raise ValueError("Selected documents do not have exactly k sentence records")

    figure_specs = [
        (
            "Original RAID dataset (before eligibility filtering and balancing)",
            "original_full_dataset",
            [
                (original_documents, "n_sentences", "Sentences per document"),
                (original_documents, "n_words", "Words per document"),
                (original_sentences, "n_words", "Words per sentence"),
            ],
        ),
        (
            f"Balanced unmatched dataset (first {k} sentences per document)",
            "balanced_first_k_dataset",
            [
                (selected_documents, "first_k_words", f"Total words in first {k} sentences"),
                (selected_documents, "mean_words_per_sentence", f"Mean words per sentence (first {k})"),
                (selected_sentences, "n_words", f"Words per individual sentence (first {k})"),
            ],
        ),
    ]

    figures = []
    for title, filename, panels in figure_specs:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
        for ax, (frame, column, xlabel) in zip(axes, panels):
            values = pd.to_numeric(frame[column], errors="coerce")
            upper = float(values.quantile(0.99))
            subset = frame.loc[values.notna() & values.le(upper)].copy()
            subset[column] = values.loc[subset.index]
            if subset.empty:
                raise ValueError(f"No observations for plot: {xlabel}")
            if subset[column].nunique() < 2:
                ax.axvline(float(subset[column].iloc[0]), color="black", lw=2, ls="--")
            else:
                sns.kdeplot(
                    data=subset, x=column, hue="source", hue_order=sources,
                    palette=palette, common_norm=False, bw_adjust=1.0,
                    cut=0, clip=(0, None), linewidth=2, fill=False,
                    warn_singular=False, legend=False, ax=ax,
                )
            ax.set(xlabel=xlabel, ylabel="Probability density", title=xlabel)
            ax.set_xlim(left=0, right=max(upper, 1))
            ax.grid(alpha=0.2)
        fig.suptitle(title, fontsize=14, fontweight="semibold")
        fig.legend(handles=handles, loc="outside lower center", ncol=min(len(sources), 7))
        if save_dir is not None:
            dest = Path(save_dir)
            dest.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest / f"{filename}.png", dpi=220, bbox_inches="tight")
        plt.show()
        figures.append(fig)
    return figures


__all__ = ["analyze_sentence_lengths", "load_or_analyze_sentence_lengths",
           "prepare_unmatched_eligible", "select_unmatched_dataset",
           "plot_unmatched_distributions"]
