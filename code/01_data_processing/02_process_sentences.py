"""RAID sentence statistics and distribution-preserving dataset selection.

Import from a notebook; no CLI or main(). Requires pandas, numpy, spaCy,
'en_core_web_sm', and code/config.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone

SELECTION_VERSION = 2  # Invalidate results from the previous selection algorithm.

import numpy as np
import pandas as pd

# Allow imports when this numbered script is loaded directly from a notebook.
import sys
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (
    CHUNK_SIZE, BATCH_SIZE, RANDOM_SEED, LENGTH_BINS, DEFAULT_N_SENTENCES,
    SPACY_MODEL, DOMAIN_NAMES, MODEL_NAMES, MODEL_ORDER,
    RESULTS_DIR, FILTERED_DIR, locate_training_file,
)


def _slug(value):
    if value is None:
        return 'all'
    if isinstance(value, (list, tuple, set)):
        value = '-'.join(sorted(map(str, value)))
    return re.sub(r'[^a-z0-9_-]+', '-', str(value).lower()).strip('-') or 'empty'


def _domain_key(domain):
    for key, label in DOMAIN_NAMES.items():
        if str(domain).lower().strip() in (key.lower(), label.lower()):
            return key
    raise ValueError(f'Unknown domain: {domain!r}')


def _run_name(meta):
    return '__'.join([
        f"domain-{_slug(meta['domain'])}",
        f"prompting-{_slug(meta['prompting'])}",
        f"attack-{_slug(meta['attack'])}",
        f"decoding-{_slug(meta['decoding'])}",
        f"rep-{_slug(meta['repetition_penalty'])}",
    ])


def _find(columns, *names):
    lookup = {str(c).lower().strip(): c for c in columns}
    return next((lookup[n.lower()] for n in names if n.lower() in lookup), None)


def _match(series, value):
    if value is None:
        return pd.Series(True, index=series.index)
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError('Empty filter collection')
    allowed = {str(x).strip().lower() for x in values}
    return series.astype('string').str.strip().str.lower().isin(allowed).fillna(False)


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


def _proportional_sample(group, target, bin_col, rng):
    """Sample exactly target rows, approximately preserving bin proportions.

    Largest-remainder allocation ensures the requested size is exact; all
    observations are sampled without replacement.
    """
    if not 0 < target <= len(group):
        raise ValueError('Target must be positive and no larger than the pool')
    if target == len(group):
        return group.copy()
    counts = group[bin_col].value_counts(sort=False).sort_index()
    ideal = counts.to_numpy(dtype=float) * target / len(group)
    allocations = np.floor(ideal).astype(int)
    remainder = target - int(allocations.sum())
    if remainder:
        fractions = ideal - allocations
        order = np.argsort(-fractions, kind='stable')
        allocations[order[:remainder]] += 1
    indices = []
    for bin_value, n_take in zip(counts.index, allocations):
        candidates = group.index[group[bin_col].eq(bin_value)].to_numpy()
        if n_take:
            indices.extend(rng.choice(candidates, size=int(n_take), replace=False))
    sampled = group.loc[indices].copy()
    if len(sampled) != target:
        raise AssertionError('Proportional sampling did not reach the target')
    return sampled


def _source_length_bins(group, n_bins):
    """Quantile bins based on this source's own first-n word lengths."""
    bins = pd.qcut(group['first_n_words'], q=n_bins,
                   labels=False, duplicates='drop')
    return bins.fillna(0).astype(int)



def _length_matched_sample(pool, sources, counts, rng):
    """Maximum-size, without-replacement length matching to the smallest source.

    Uses minimum-cost bipartite assignment on absolute word-count differences.
    Exact equality of distributions is not guaranteed when supports differ.
    """
    from scipy.optimize import linear_sum_assignment

    target = min(int(counts[source]) for source in sources)
    reference = next(source for source in sources if int(counts[source]) == target)
    reference_rows = pool.loc[pool['source'].eq(reference)].copy()
    reference_lengths = reference_rows['first_n_words'].to_numpy(dtype=float)
    pieces = [reference_rows]

    for source in sources:
        if source == reference:
            continue
        candidates = pool.loc[pool['source'].eq(source)].copy()
        lengths = candidates['first_n_words'].to_numpy(dtype=float)
        # Randomize only the order of equally good candidates, reproducibly.
        order = rng.permutation(len(candidates))
        candidates = candidates.iloc[order]
        lengths = lengths[order]
        cost = np.abs(reference_lengths[:, None] - lengths[None, :])
        row_idx, col_idx = linear_sum_assignment(cost)
        pieces.append(candidates.iloc[col_idx].copy())

    return pd.concat(pieces, ignore_index=True)


def _select_at_threshold(documents, sentences, n, sources, *, match_source_id,
                         sampling, n_bins, seed):
    """Construct a balanced candidate at a fixed sentence threshold."""
    eligible = documents.loc[
        documents['n_sentences'].ge(n) & documents['source'].isin(sources)
    ].copy()
    counts = eligible['source'].value_counts()
    if any(counts.get(source, 0) == 0 for source in sources):
        raise ValueError('At least one requested source has no eligible documents')

    # Reconstruct exactly the first n sentences, not the full document.
    first = sentences.loc[
        sentences['sentence_idx'].le(n) & sentences['id'].isin(eligible['id']),
        ['id', 'sentence_idx', 'n_words', 'sentence_text']
    ].sort_values(['id', 'sentence_idx'])
    aggregates = first.groupby('id', sort=False).agg(
        first_n_words=('n_words', 'sum'),
        first_n_text=('sentence_text', ' '.join),
        counted=('sentence_idx', 'size'),
    )
    eligible = eligible.join(aggregates, on='id')
    if eligible['counted'].ne(n).any():
        raise ValueError('Missing sentence records for eligible documents')
    pool = eligible.drop(columns='counted')
    rng = np.random.default_rng(seed)

    if match_source_id:
        pool = pool.dropna(subset=['source_id']).copy()
        pool = pool.loc[pool['source_id'].astype('string').str.strip().ne('')]
        pool['_random'] = rng.random(len(pool))
        pool = (pool.sort_values('_random')
                .drop_duplicates(['source_id', 'source'])
                .drop(columns='_random'))
        availability = pool.groupby(['source_id', 'source']).size().unstack(fill_value=0)
        availability = availability.reindex(columns=sources, fill_value=0)
        common_ids = availability.index[availability.gt(0).all(axis=1)]
        if len(common_ids) == 0:
            raise ValueError('No complete source_id groups')
        pool = pool.loc[pool['source_id'].isin(common_ids)].copy()
        # In paired mode all complete groups are retained. A length bin is
        # descriptive; sampling must not break source_id alignment.
        wide = pool.pivot(index='source_id', columns='source', values='first_n_words')
        mean_length = wide[sources].mean(axis=1)
        bins = pd.qcut(mean_length, q=n_bins, labels=False, duplicates='drop')
        pool = pool.join(bins.rename('length_bin'), on='source_id')
        selected = pool
    else:
        # The requested target is determined by LLMs only, not by Human.
        llm_sources = [source for source in sources if source != 'Human']
        target = min(int(counts[source]) for source in llm_sources)
        if any(int(counts[source]) < target for source in sources):
            raise ValueError(
                f'An included source has fewer than {target} eligible documents; '
                'cannot balance without replacement'
            )

        if sampling == 'proportional':
            pieces = []
            for source in sources:
                source_pool = pool.loc[pool['source'].eq(source)].copy()
                # Each source is stratified against its OWN distribution.
                source_pool['length_bin'] = _source_length_bins(
                    source_pool, n_bins
                )
                pieces.append(_proportional_sample(
                    source_pool, target, 'length_bin', rng
                ))
            selected = pd.concat(pieces, ignore_index=True)
        elif sampling == 'length_match':
            # Kept for backwards compatibility; unlike proportional,
            # this deliberately makes source length distributions similar.
            pool['length_bin'] = pd.qcut(
                pool['first_n_words'], q=n_bins,
                labels=False, duplicates='drop'
            ).fillna(0).astype(int)
            selected = _length_matched_sample(pool, sources, counts, rng)
        else:
            # Historical common-bin sampling, which may discard observations
            # beyond the global minimum to equalize bin frequencies.
            pool['length_bin'] = pd.qcut(
                pool['first_n_words'], q=n_bins,
                labels=False, duplicates='drop'
            ).fillna(0).astype(int)
            bin_counts = pool.groupby(['length_bin', 'source']).size().unstack(fill_value=0)
            capacities = bin_counts.reindex(columns=sources, fill_value=0).min(axis=1)
            if int(capacities.sum()) == 0:
                raise ValueError('No common length-bin capacity')
            pieces = []
            for source in sources:
                source_pool = pool.loc[pool['source'].eq(source)]
                for bin_id, quota in capacities.items():
                    if int(quota) == 0:
                        continue
                    choices = source_pool.index[source_pool['length_bin'].eq(bin_id)].to_numpy()
                    chosen = rng.choice(choices, size=int(quota), replace=False)
                    pieces.append(pool.loc[chosen])
            selected = pd.concat(pieces, ignore_index=True)

        if sampling == 'proportional':
            final_counts = selected['source'].value_counts()
            if any(int(final_counts.get(source, 0)) != target for source in sources):
                raise AssertionError('Unmatched proportional selection is not balanced')

    selected = selected.sort_values(['source', 'id']).reset_index(drop=True)
    selected_counts = selected['source'].value_counts()
    return selected, counts, selected_counts


def select_sentence_dataset(
    documents, sentences, min_sentences=5, max_sentences=10, *,
    domain=None, prompting=None, attack='null', decoding='greedy',
    repetition_penalty='no', match_source_id=False, seed=RANDOM_SEED,
    n_bins=LENGTH_BINS, save=True, include_human=True, output_dir=None,
    sampling='proportional',
):
    """Find the best first-n-sentence balanced dataset across an inclusive range.

    Score each n by the *minimum number of selected documents among LLMs*,
    after matching and length stratification. On ties, prefer larger n.
    Human is included in balancing when include_human=True, but is not used
    as a separate scoring model. Return (selected, threshold_comparison).

    In unmatched proportional mode, select the minimum eligible LLM count
    from every source while preserving each source-specific length distribution
    approximately via quantile-stratified sampling without replacement.
    Human, if included, must have at least that many eligible documents.
    The legacy length_match and common_bins modes remain available.
    In paired mode retain all complete source_id groups, one record per
    (source_id, source); sampling mode does not change those groups.
    """
    if not isinstance(min_sentences, int) or not isinstance(max_sentences, int):
        raise TypeError('min_sentences and max_sentences must be integers')
    if not 1 <= min_sentences <= max_sentences:
        raise ValueError('Require 1 <= min_sentences <= max_sentences')
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    if sampling not in {'common_bins', 'proportional', 'length_match'}:
        raise ValueError("sampling must be 'common_bins', 'proportional' or 'length_match'")
    if documents['id'].duplicated().any():
        raise ValueError('Duplicate document IDs')
    if sentences.duplicated(['id', 'sentence_idx']).any():
        raise ValueError('Duplicate sentence IDs')

    meta = dict(documents.attrs.get('raid_metadata', {}))
    requested = dict(domain=domain, prompting=prompting, attack=attack,
                     decoding=decoding, repetition_penalty=repetition_penalty)
    if domain is not None:
        requested['domain'] = _domain_key(domain)
    for key, value in requested.items():
        if key in meta and value is not None and meta[key] != value:
            raise ValueError(
                f'Filter {key}={value!r} differs from analysis ({meta[key]!r}); '
                'rerun analyze_sentence_lengths'
            )
    if not meta:
        meta = requested
    # Fill metadata keys required for output path names, even when caller
    # loaded documents from CSV and DataFrame.attrs were not preserved.
    for key, value in requested.items():
        meta.setdefault(key, value)
    if meta['domain'] is None:
        raise ValueError('Provide domain when documents metadata is unavailable')

    llms = [s for s in MODEL_ORDER if s != 'Human']
    sources = (['Human'] if include_human else []) + llms
    if not set(sources).issubset(set(documents['source'].dropna())):
        missing = sorted(set(sources) - set(documents['source'].dropna()))
        raise ValueError(f'Missing sources: {missing}')

    candidates = []
    report_rows = []
    for n in range(min_sentences, max_sentences + 1):
        eligible_counts = documents.loc[
            documents['n_sentences'].ge(n), 'source'
        ].value_counts()
        row = {
            'n_sentences': n,
            'min_eligible_llm': min(int(eligible_counts.get(s, 0)) for s in llms),
        }
        for source in sources:
            row[f'eligible_{source}'] = int(eligible_counts.get(source, 0))
        try:
            candidate, _, selected_counts = _select_at_threshold(
                documents, sentences, n, sources,
                match_source_id=match_source_id, sampling=sampling,
                n_bins=n_bins, seed=seed,
            )
            row['min_selected_llm'] = min(int(selected_counts.get(s, 0)) for s in llms)
            row['total_selected'] = len(candidate)
            row['status'] = 'ok'
            for source in sources:
                row[f'selected_{source}'] = int(selected_counts.get(source, 0))
            candidates.append((row['min_selected_llm'], n, candidate))
        except ValueError as exc:
            row.update(min_selected_llm=0, total_selected=0, status=str(exc))
            for source in sources:
                row[f'selected_{source}'] = 0
        report_rows.append(row)

    if not candidates:
        raise ValueError('No valid dataset in the requested sentence range')

    # Primary criterion: largest minimum LLM count. Tie-break: largest n.
    best_count, best_n, selected = max(candidates, key=lambda item: (item[0], item[1]))
    comparison = pd.DataFrame(report_rows)
    comparison['is_best'] = comparison['n_sentences'].eq(best_n)
    result_meta = dict(meta)
    result_meta.update(
        min_sentences=min_sentences, max_sentences=max_sentences,
        best_n_sentences=best_n, best_min_llm_documents=best_count,
        match_source_id=match_source_id, seed=seed, n_bins=n_bins,
        sampling=sampling, include_human=include_human,
        selection_version=SELECTION_VERSION,
        selection_criterion='maximize minimum selected LLM documents; tie: larger n',
        selected_counts={s: int(selected['source'].eq(s).sum()) for s in sources},
    )
    selected.attrs['raid_metadata'] = result_meta
    comparison.attrs['raid_metadata'] = result_meta

    if save:
        base = Path(output_dir) if output_dir is not None else FILTERED_DIR
        folder = (base / _run_name(meta) /
                  f'range-{min_sentences}-{max_sentences}__best-{best_n}'
                  f'__matched-{int(match_source_id)}__{sampling}'
                  f'__seed-{seed}__bins-{n_bins}__human-{int(include_human)}')
        _save_metadata(folder, result_meta)
        selected.to_csv(folder / 'selected_documents.csv', index=False)
        comparison.to_csv(folder / 'threshold_comparison.csv', index=False)
        selected_row = comparison.loc[comparison['is_best']].iloc[0]
        pd.DataFrame([
            {'source': s, 'eligible': int(selected_row[f'eligible_{s}']),
             'selected': int(selected_row[f'selected_{s}'])}
            for s in sources
        ]).to_csv(folder / 'sampling_report.csv', index=False)
        print('Saved selected dataset:', folder)

    print(f'Best n={best_n}; minimum selected LLM documents={best_count:,}')
    return selected, comparison


def load_or_select_sentence_dataset(documents, sentences, *, min_sentences=5,
                                    max_sentences=10, domain=None, prompting=None,
                                    attack='null', decoding='greedy',
                                    repetition_penalty='no', match_source_id=False,
                                    sampling='proportional', n_bins=LENGTH_BINS,
                                    seed=RANDOM_SEED, include_human=True,
                                    output_dir=None):
    """Load a compatible selection or compute it using the original output layout."""
    base = Path(output_dir) if output_dir is not None else FILTERED_DIR
    filters = dict(domain=_domain_key(domain), prompting=prompting, attack=attack,
                   decoding=decoding, repetition_penalty=repetition_penalty)
    root = base / _run_name(filters)
    expected = dict(min_sentences=min_sentences, max_sentences=max_sentences,
                    match_source_id=bool(match_source_id), sampling=sampling,
                    seed=seed, n_bins=n_bins, include_human=bool(include_human),
                    selection_version=SELECTION_VERSION)
    pattern = (f'range-{min_sentences}-{max_sentences}__best-*'
               f'__matched-{int(match_source_id)}__{sampling}'
               f'__seed-{seed}__bins-{n_bins}__human-{int(include_human)}')
    matches = []
    for folder in root.glob(pattern):
        required = ('metadata.json', 'selected_documents.csv',
                    'threshold_comparison.csv', 'sampling_report.csv')
        if not all((folder / name).is_file() for name in required):
            continue
        with (folder / 'metadata.json').open(encoding='utf-8') as f:
            meta = json.load(f)
        if all(meta.get(k) == v for k, v in {**filters, **expected}.items()):
            matches.append((folder, meta))
    if len(matches) > 1:
        raise RuntimeError(f'Multiple compatible selections: {[str(x[0]) for x in matches]}')
    if matches:
        folder, meta = matches[0]
        selected = pd.read_csv(folder / 'selected_documents.csv',
                               dtype={'id': 'string', 'source_id': 'string'})
        comparison = pd.read_csv(folder / 'threshold_comparison.csv')
        selected.attrs['raid_metadata'] = meta
        comparison.attrs['raid_metadata'] = meta
        print('Loaded selection:', folder)
        return selected, comparison
    return select_sentence_dataset(
        documents, sentences, min_sentences=min_sentences,
        max_sentences=max_sentences, domain=domain, prompting=prompting,
        attack=attack, decoding=decoding,
        repetition_penalty=repetition_penalty, match_source_id=match_source_id,
        sampling=sampling, n_bins=n_bins, seed=seed,
        include_human=include_human, output_dir=output_dir, save=True,
    )


def original_first_n_lengths(documents, sentences, n):
    """Word counts in the first n sentences for every eligible document."""
    eligible = documents.loc[documents['n_sentences'].ge(n), ['id', 'source']]
    first = sentences.loc[sentences['sentence_idx'].le(n),
                          ['id', 'sentence_idx', 'n_words']]
    aggregated = first.groupby('id', as_index=False).agg(
        first_n_words=('n_words', 'sum'), counted=('sentence_idx', 'size'))
    result = eligible.merge(aggregated, on='id', how='left', validate='one_to_one')
    if not result['counted'].eq(n).all():
        raise ValueError(f'Incomplete sentence records for n={n}')
    return result.drop(columns='counted')


def selection_diagnostics(documents, sentences, selections, *, plot=True):
    """Validate selected counts and display original-vs-selected length KDEs.

    selections: mapping of label -> (selected, comparison).
    Figures are displayed only; nothing is saved.
    Returns (count_table, distribution_table).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wasserstein_distance
    from matplotlib.lines import Line2D
    from IPython.display import display

    counts_rows, dist_rows = [], []
    sources = [s for s in MODEL_ORDER if s in documents['source'].unique()]
    llms = [s for s in sources if s != 'Human']
    if not llms:
        raise ValueError('No LLM sources')
    prepared = []
    for label, (selected, comparison) in selections.items():
        meta = selected.attrs['raid_metadata']
        n = int(meta['best_n_sentences'])
        included = [s for s in sources if meta.get('include_human', True) or s != 'Human']
        before = original_first_n_lengths(documents, sentences, n)
        after = selected[['source', 'first_n_words']]
        original_counts = before['source'].value_counts().reindex(included, fill_value=0)
        selected_counts = after['source'].value_counts().reindex(included, fill_value=0)
        target = min(int(original_counts[s]) for s in llms)
        if selected['id'].duplicated().any() or not selected['n_sentences'].ge(n).all():
            raise AssertionError(f'{label}: duplicate IDs or invalid sentence threshold')
        if selected_counts.nunique() != 1:
            raise AssertionError(f'{label}: unequal source counts: {selected_counts.to_dict()}')
        if not meta['match_source_id']:
            if not selected_counts.eq(target).all():
                raise AssertionError(f'{label}: expected {target} per source')
        else:
            if selected['source_id'].isna().any() or selected.duplicated(['source_id', 'source']).any():
                raise AssertionError(f'{label}: invalid source_id groups')
            if not selected.groupby('source_id')['source'].nunique().eq(len(included)).all():
                raise AssertionError(f'{label}: incomplete source_id groups')
        for s in included:
            original_values = before.loc[before['source'].eq(s), 'first_n_words'].dropna().to_numpy(float)
            sampled_values = after.loc[after['source'].eq(s), 'first_n_words'].dropna().to_numpy(float)
            counts_rows.append(dict(selection=label, source=s, best_n=n,
                                    eligible=int(original_counts[s]), selected=int(selected_counts[s])))
            dist_rows.append(dict(selection=label, source=s,
                                  median_before=float(np.median(original_values)),
                                  median_after=float(np.median(sampled_values)),
                                  wasserstein=float(wasserstein_distance(original_values, sampled_values))))
        prepared.append((label, n, before, after, included))
        print(f'{label}: n={n}, {int(selected_counts.iloc[0]):,} documents per source')
    count_table = pd.DataFrame(counts_rows)
    distribution_table = pd.DataFrame(dist_rows)
    display(count_table)
    display(distribution_table.round(2))
    if plot:
        palette = dict(zip(sources, sns.color_palette('colorblind', n_colors=len(sources))))
        # The original dataset is shared whenever the two selections use the same n.
        # If best_n differs, separate original panels are necessary for a valid comparison.
        original_by_n = {}
        panels = []
        for label, n, before, after, included in prepared:
            if n not in original_by_n:
                original_by_n[n] = (before, included)
                panels.append((f'Original dataset', n, before, included))
            panels.append((f'{label} dataset', n, after, included))

        fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4.8),
                                 squeeze=False, constrained_layout=True)
        axes = axes.ravel()
        # Use identical x limits across panels sharing the same sentence threshold.
        xlimits = {}
        for _, n, frame, _ in panels:
            value = float(frame['first_n_words'].quantile(.995))
            xlimits[n] = max(xlimits.get(n, 0.0), value)

        for ax, (title, n, frame, included) in zip(axes, panels):
            counts = frame['source'].value_counts().reindex(included, fill_value=0)
            for source in included:
                values = frame.loc[frame['source'].eq(source), 'first_n_words'].dropna()
                if len(values) > 1 and values.nunique() > 1:
                    sns.kdeplot(x=values, color=palette[source], bw_adjust=1.2,
                                cut=0, clip=(0, None), linewidth=2, ax=ax,
                                label=f'{source} (N={int(counts[source]):,})')
            count_text = (f'N={int(counts.iloc[0]):,} per source'
                          if counts.nunique() == 1 else
                          f'N={int(counts.sum()):,} total')
            ax.set(title=f'{title}\n{count_text} | n={n}',
                   xlabel=f'Words in first {n} sentences', ylabel='Density')
            ax.set_xlim(0, xlimits[n])
            ax.grid(False, which='both', axis='both')
            ax.minorticks_off()
            ax.legend(title='Source (documents)', fontsize=8, title_fontsize=9,
                      frameon=False, loc='best')
        plt.show()
    return count_table, distribution_table



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