"""RAID sentence statistics and unpaired, length-stratified dataset selection.

Import from a notebook; no CLI or main(). Requires pandas, numpy, spaCy,
'en_core_web_sm', and raid_config.py in the same code directory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone

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
    """Allocate target observations across bins by largest remainders."""
    if target > len(group):
        raise ValueError('Cannot sample more than available without replacement')
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
        candidates = group.index[group[bin_col] == bin_value].to_numpy()
        if n_take:
            indices.extend(rng.choice(candidates, size=n_take, replace=False).tolist())
    return group.loc[indices].copy()


def select_sentence_dataset(
    documents, sentences, n_sentences=DEFAULT_N_SENTENCES, *, domain=None, prompting=None,
    attack="null", decoding="greedy", repetition_penalty="no",
    match_source_id=False, seed=RANDOM_SEED, n_bins=LENGTH_BINS, save=True,
    include_human=True, output_dir=None, sampling="common_bins",
):
    """Select first n sentences with equal source counts.

    match_source_id=True: sample complete groups of identical source_id.
    match_source_id=False: sample independent documents per source.
    sampling='common_bins' seeks common length-bin quotas across sources;
    'proportional' preserves each source's original length distribution.
    """
    if n_sentences < 1 or n_bins < 1:
        raise ValueError("n_sentences and n_bins must be positive")
    if sampling not in {"common_bins", "proportional"}:
        raise ValueError("sampling must be common_bins or proportional")
    if documents['id'].duplicated().any() or sentences.duplicated(['id', 'sentence_idx']).any():
        raise ValueError("Duplicate document or sentence IDs")
    meta = dict(documents.attrs.get("raid_metadata", {}))
    requested = dict(domain=domain, prompting=prompting, attack=attack,
                     decoding=decoding, repetition_penalty=repetition_penalty)
    if requested['domain'] is not None:
        requested['domain'] = _domain_key(requested['domain'])
    for key, value in requested.items():
        if key in meta and value is not None and meta[key] != value:
            raise ValueError(f"Filter {key}={value!r} differs from analysis ({meta[key]!r}); rerun analysis")
    if not meta:
        meta = requested
    eligible = documents.loc[documents['n_sentences'].ge(n_sentences)].copy()
    first = sentences.loc[sentences['sentence_idx'].le(n_sentences) &
                          sentences['id'].isin(eligible['id'])].sort_values(['id','sentence_idx'])
    aggregates = first.groupby('id').agg(first_n_words=('n_words', 'sum'),
        first_n_text=('sentence_text', ' '.join), counted=('sentence_idx','count'))
    eligible = eligible.join(aggregates, on='id')
    if eligible['counted'].ne(n_sentences).any():
        raise ValueError("Sentence records missing for eligible documents")
    eligible = eligible.drop(columns='counted')
    sources = (['Human'] if include_human else []) + [x for x in MODEL_ORDER if x != 'Human']
    counts = eligible['source'].value_counts()
    missing = [s for s in sources if counts.get(s, 0) == 0]
    if missing:
        raise ValueError(f"No eligible documents for: {missing}")
    pool = eligible.loc[eligible['source'].isin(sources)].copy()
    rng = np.random.default_rng(seed)
    if match_source_id:
        # Only groups with at least one document from EVERY source.
        pool = pool.dropna(subset=['source_id'])
        groups = pool.groupby(['source_id','source'], sort=False).size().unstack(fill_value=0)
        common_ids = groups.index[groups.reindex(columns=sources, fill_value=0).gt(0).all(axis=1)]
        if len(common_ids) == 0:
            raise ValueError("No complete source_id groups for requested sources")
        pool = pool[pool['source_id'].isin(common_ids)].copy()
        # One record per source and source_id, deterministic random selection.
        pool['_random'] = rng.random(len(pool))
        pool = (pool.sort_values('_random').drop_duplicates(['source_id','source'])
                    .drop(columns='_random'))
        wide = pool.pivot(index='source_id', columns='source', values='first_n_words')
        wide['length'] = wide[sources].mean(axis=1)
        wide['length_bin'] = pd.qcut(wide['length'], q=n_bins, labels=False, duplicates='drop')
        if sampling == 'proportional':
            chosen = wide.index.to_numpy()
        else:
            chosen = wide.index.to_numpy()  # all complete pairs; no arbitrary downsampling
        selected = pool[pool['source_id'].isin(chosen)].copy()
        selected = selected.join(wide['length_bin'], on='source_id')
        target = len(chosen)
    else:
        pool['length_bin'] = pd.qcut(pool['first_n_words'], q=n_bins, labels=False, duplicates='drop')
        if sampling == 'proportional':
            target = min(int(counts[s]) for s in sources)
            selected = pd.concat([_proportional_sample(pool[pool.source.eq(s)], target, 'length_bin', rng)
                                  for s in sources], ignore_index=True)
        else:
            # Equal quota per bin, bounded by the smallest source count in each bin.
            bin_counts = pool.groupby(['length_bin','source']).size().unstack(fill_value=0)
            capacity = bin_counts.reindex(columns=sources, fill_value=0).min(axis=1)
            target = int(capacity.sum())
            if target == 0:
                raise ValueError("No overlapping length bins across all sources")
            pieces = []
            for source in sources:
                for bin_id, quota in capacity.items():
                    if quota:
                        candidates = pool[(pool.source == source) & (pool.length_bin == bin_id)]
                        chosen_idx = rng.choice(candidates.index.to_numpy(), size=int(quota), replace=False)
                        pieces.append(pool.loc[chosen_idx])
            selected = pd.concat(pieces, ignore_index=True)
    selected = selected.sort_values(['source','id']).reset_index(drop=True)
    result_meta = dict(meta)
    result_meta.update(n_sentences=n_sentences, match_source_id=match_source_id,
        seed=seed, n_bins=n_bins, sampling=sampling, include_human=include_human,
        n_per_source=target, eligible_counts={s:int(counts.get(s,0)) for s in sources},
        selected_counts={s:int((selected.source == s).sum()) for s in sources})
    selected.attrs['raid_metadata'] = result_meta
    if save:
        base = Path(output_dir) if output_dir is not None else FILTERED_DIR
        folder = base / _run_name(meta) / f"n-{n_sentences}__matched-{int(match_source_id)}__{sampling}__seed-{seed}__bins-{n_bins}"
        _save_metadata(folder, result_meta)
        selected.to_csv(folder/'selected_documents.csv', index=False)
        pd.DataFrame({'source':sources, 'eligible':[int(counts.get(s,0)) for s in sources],
            'selected':[int((selected.source == s).sum()) for s in sources]}).to_csv(folder/'sampling_report.csv', index=False)
        print('Saved selected dataset:', folder)
    return selected
