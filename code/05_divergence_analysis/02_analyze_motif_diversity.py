"""Document-level motif entropy and HHI, with document bootstrap.

Uses the exact no-wildcard motif-by-document frequency table created by
01_analyze_motif_divergence.py. Outputs into the SAME experiment directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = '1.0-document-diversity-bootstrap'
METRICS = ('entropy_bits', 'hhi')


def _file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _save(fig, folder, stem, show):
    fig.savefig(folder / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(folder / (stem + '.png'), dpi=240, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def _mean_draws(values, n_boot, rng, batch=500):
    n = len(values)
    draws = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, batch):
        end = min(n_boot, start + batch)
        indices = rng.integers(n, size=(end-start, n))
        draws[start:end] = values[indices].mean(axis=1)
    return draws


def _title(genre, n_sentences, counts):
    if genre is None:
        genre = 'RST'
    name = str(genre).replace('_', ' ').title()
    if str(genre).lower() == 'reddit':
        name += ' posts'
    elif str(genre).lower() == 'news':
        name += ' articles'
    elif str(genre).lower() in ('abstract', 'abstracts'):
        name += ' abstracts'
    ntext = (f'{counts.iloc[0]} samples per source' if counts.nunique() == 1
             else f'{counts.sum()} samples across {len(counts)} sources')
    sent = f' with {n_sentences} sentences' if n_sentences is not None else ''
    return f'Document-level Motif Diversity of {name}{sent} and {ntext}'


def analyze_motif_diversity(
    *, divergence_dir=None, results_dir=None, dataset_name=None,
    genre=None, n_sentences=None, human_source='Human',
    n_boot=10000, confidence=0.95, seed=42, rebuild=False, show=True,
):
    """Analyze entropy and HHI of individual RST documents.

    Pass divergence_dir directly (recommended), or results_dir + dataset_name.
    Returns (experiment_directory, result_tables).

    Entropy and HHI are undefined for documents with zero motif occurrences.
    Those documents are retained with NaN values and excluded from mean/bootstrap.
    The effective sample sizes and excluded-document counts are reported.
    """
    if divergence_dir is None:
        if results_dir is None or dataset_name is None:
            raise ValueError('Pass divergence_dir or both results_dir and dataset_name')
        divergence_dir = Path(results_dir) / 'divergence_analysis' / dataset_name
    out = Path(divergence_dir)
    input_csv = out / 'motif_document_frequencies.csv'
    if not input_csv.exists():
        raise FileNotFoundError(f'Run motif divergence analysis first: {input_csv}')
    if not 0 < confidence < 1 or n_boot < 100:
        raise ValueError('Require 0 < confidence < 1 and n_boot >= 100')
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    csv_names = ('motif_document_diversity', 'motif_diversity_summary',
                 'motif_diversity_bootstrap_draws', 'motif_diversity_deltas',
                 'motif_diversity_delta_draws')
    fig_names = ('05_document_entropy_boxplot', '06_document_hhi_boxplot',
                 '07_document_diversity_bootstrap')
    manifest_path = out / 'motif_diversity_manifest.json'
    settings = dict(version=VERSION, input_hash=_file_hash(input_csv),
                    human_source=human_source, n_boot=int(n_boot),
                    confidence=float(confidence), seed=int(seed),
                    genre=genre, n_sentences=n_sentences)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    ready = (manifest_path.exists()
             and all((out / (n + '.csv')).exists() for n in csv_names)
             and all((fig_dir / (n + '.' + ext)).exists()
                     for n in fig_names for ext in ('pdf', 'png')))
    if ready and not rebuild:
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old.get('fingerprint') == fingerprint:
            tables = {name: pd.read_csv(out / (name + '.csv')) for name in csv_names}
            if show:
                from IPython.display import Image, display
                for stem in fig_names:
                    display(Image(filename=str(fig_dir / (stem + '.png'))))
            print(f'Loaded document-level motif diversity cache: {out}')
            return out, tables

    frame = pd.read_csv(input_csv, dtype={'source': str, 'document_id': str,
                                          'motif_id': str})
    required = {'source', 'document_id', 'motif_id', 'frequency'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    if frame.duplicated(['source', 'document_id', 'motif_id']).any():
        raise ValueError('Duplicate source/document/motif entries')
    if frame['frequency'].isna().any() or (frame['frequency'] < 0).any():
        raise ValueError('Motif frequencies must be nonnegative and nonmissing')
    if not np.all(np.isclose(frame.frequency, np.round(frame.frequency))):
        raise ValueError('Motif frequencies must be integer counts')
    if not len(frame):
        raise ValueError('Empty motif-document frequency table')

    # Full global motif vocabulary, shared by all sources; missing combinations
    # are zero-filled as a safeguard, even though the upstream table is dense.
    matrix = frame.pivot(index=['source', 'document_id'],
                         columns='motif_id', values='frequency').fillna(0)
    values = matrix.to_numpy(dtype=np.float64)
    total = values.sum(axis=1)
    active = total > 0
    probabilities = np.divide(values, total[:, None],
                              out=np.zeros_like(values), where=total[:, None] > 0)
    log_p = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_p, where=probabilities > 0)
    entropy = -(probabilities * log_p).sum(axis=1)
    hhi = np.square(probabilities).sum(axis=1)
    richness = np.count_nonzero(values, axis=1)
    entropy[~active] = np.nan
    hhi[~active] = np.nan
    doc = pd.DataFrame({
        'source': matrix.index.get_level_values('source'),
        'document_id': matrix.index.get_level_values('document_id'),
        'n_motif_occurrences': total.astype(np.int64),
        'motif_richness': richness.astype(np.int64),
        'entropy_bits': entropy,
        'hhi': hhi,
        'effective_motifs_shannon': np.where(active, np.exp2(np.nan_to_num(entropy)), np.nan),
        'effective_motifs_simpson': np.divide(1.0, hhi, out=np.full(len(hhi), np.nan), where=hhi > 0),
    })
    sources = sorted(doc.source.unique())
    if human_source not in sources:
        raise ValueError(f'Human reference not found: {human_source}')
    sources.remove(human_source)
    sources.insert(0, human_source)
    counts = doc.groupby('source').size().reindex(sources)
    title = _title(genre, n_sentences, counts)
    rng = np.random.default_rng(seed)
    alpha = (1 - confidence) / 2
    summary_rows, draw_rows, delta_rows, delta_draw_rows = [], [], [], []
    draws_by_metric = {}
    for metric in METRICS:
        for source in sources:
            group = doc.loc[doc.source == source, metric].dropna().to_numpy(float)
            n = len(group)
            if n < 2:
                raise ValueError(f'Need at least two nonempty documents for {source}: {metric}')
            draws = _mean_draws(group, n_boot, rng)
            draws_by_metric[(source, metric)] = draws
            lo, hi = np.quantile(draws, [alpha, 1-alpha])
            summary_rows.append(dict(source=source, metric=metric,
                                     n_documents_total=int(counts[source]),
                                     n_documents_valid=n,
                                     n_documents_empty=int(counts[source]-n),
                                     mean=float(group.mean()),
                                     sd=float(group.std(ddof=1)),
                                     sem=float(group.std(ddof=1)/np.sqrt(n)),
                                     median=float(np.median(group)),
                                     ci_low=float(lo), ci_high=float(hi),
                                     n_boot=n_boot, confidence=confidence))
            draw_rows.extend((source, metric, i, float(x)) for i, x in enumerate(draws))
        for source in sources[1:]:
            differences = draws_by_metric[(source, metric)] - draws_by_metric[(human_source, metric)]
            point = (doc.loc[doc.source == source, metric].mean()
                     - doc.loc[doc.source == human_source, metric].mean())
            lo, hi = np.quantile(differences, [alpha, 1-alpha])
            delta_rows.append(dict(source=source, reference=human_source,
                                   metric=metric, delta=float(point),
                                   ci_low=float(lo), ci_high=float(hi),
                                   excludes_zero=bool(lo > 0 or hi < 0),
                                   n_boot=n_boot, confidence=confidence))
            delta_draw_rows.extend((source, human_source, metric, i, float(x))
                                   for i, x in enumerate(differences))

    summary = pd.DataFrame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    tables = {
        'motif_document_diversity': doc,
        'motif_diversity_summary': summary,
        'motif_diversity_bootstrap_draws': pd.DataFrame(
            draw_rows, columns=['source', 'metric', 'iteration', 'bootstrap_mean']),
        'motif_diversity_deltas': deltas,
        'motif_diversity_delta_draws': pd.DataFrame(
            delta_draw_rows, columns=['source', 'reference', 'metric',
                                      'iteration', 'bootstrap_delta']),
    }
    palette = dict(zip(sources, plt.get_cmap('tab10').colors[:len(sources)]))
    for metric, stem, ylabel in (
        ('entropy_bits', fig_names[0], 'Shannon entropy (bits)'),
        ('hhi', fig_names[1], 'Herfindahl–Hirschman index (HHI)'),
    ):
        fig, ax = plt.subplots(figsize=(max(8, len(sources)*1.45), 5), layout='constrained')
        bp = ax.boxplot([doc.loc[doc.source == s, metric].dropna() for s in sources],
                        patch_artist=True, labels=sources, showfliers=True,
                        medianprops={'color': 'black'})
        for patch, source in zip(bp['boxes'], sources):
            patch.set_facecolor(palette[source]); patch.set_alpha(.6)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=25)
        ax.grid(axis='y', alpha=.18)
        ax.set_title(title, fontsize=12)
        _save(fig, fig_dir, stem, show)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, .65*len(sources)+2)),
                             layout='constrained')
    for ax, metric, xlabel in zip(axes, METRICS,
                                  ('Mean document entropy (bits)', 'Mean document HHI')):
        sub = summary.set_index(['source', 'metric'])
        for i, source in enumerate(sources):
            r = sub.loc[(source, metric)]
            ax.errorbar(r['mean'], i,
                        xerr=[[r['mean']-r['ci_low']], [r['ci_high']-r['mean']]],
                        fmt='o', capsize=4, color=palette[source], elinewidth=1.7)
        ax.set_yticks(range(len(sources)), sources)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=.2)
        ax.margins(x=.12, y=.15)
    fig.suptitle(title, fontsize=12)
    _save(fig, fig_dir, fig_names[2], show)

    # Write results after all calculations and figures succeed.
    for name, table in tables.items():
        table.to_csv(out / (name + '.csv'), index=False)
    tmp = manifest_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'fingerprint': fingerprint, 'settings': settings}, indent=2),
                   encoding='utf-8')
    tmp.replace(manifest_path)
    print(f'Document-level diversity saved to: {out}')
    return out, tables
