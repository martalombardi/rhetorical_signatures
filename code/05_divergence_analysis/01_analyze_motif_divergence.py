"""Global RST motif vocabulary, exhaustive per-document recount, and JS divergence.

Motif candidates come from TreeRePair extraction; occurrences are independently
recounted on every original RST document, including across source boundaries.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VERSION = '1.4-compact-rank-scatter-js-tables-only'


def _safe(x):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', str(x)).strip('-')


def _metadata(path):
    d = {}
    for part in path.stem.split('__'):
        for key in ('genre', 'source', 'n_sentences'):
            if part.startswith(key + '-'):
                d[key] = part[len(key)+1:]
    return d


def _parse_motif(s):
    """Parse canonical motif expressions such as NS_ELABORATION(EDU,X)."""
    tokens = re.findall(r'[^(),\s]+|[(),]', s)
    def read(i):
        if i >= len(tokens) or tokens[i] in (',', '(', ')'):
            raise ValueError(f'Invalid motif: {s}')
        label = tokens[i]; i += 1
        children = []
        if i < len(tokens) and tokens[i] == '(':
            i += 1
            while True:
                child, i = read(i)
                children.append(child)
                if i >= len(tokens):
                    raise ValueError(f'Unclosed motif: {s}')
                if tokens[i] == ')':
                    i += 1
                    break
                if tokens[i] != ',':
                    raise ValueError(f'Invalid motif separator: {s}')
                i += 1
        return (label, tuple(children)), i
    result, end = read(0)
    if end != len(tokens):
        raise ValueError(f'Trailing tokens: {s}')
    return result


def _canonical_text(tree):
    label, children = tree
    return label if not children else label + '(' + ','.join(map(_canonical_text, children)) + ')'


def _rst_node(el):
    if el.tag == 'EDU':
        if len(el):
            raise ValueError('EDU cannot have children')
        return ('EDU', ())
    if not el.tag.startswith('R_') or len(el) != 2:
        raise ValueError(f'Invalid RST node: {el.tag}, children={len(el)}')
    rel, sep, nuc = el.tag[2:].rpartition('_')
    if not sep or nuc not in ('NN', 'NS', 'SN'):
        raise ValueError(f'Invalid RST relation: {el.tag}')
    return (nuc + '_' + rel.upper(), tuple(_rst_node(c) for c in el))


def _matches(pattern, node):
    if pattern[0] == 'X':
        return True
    return (pattern[0] == node[0] and len(pattern[1]) == len(node[1])
            and all(_matches(a, b) for a, b in zip(pattern[1], node[1])))


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node[1])


def _js(p, q):
    """Jensen-Shannon divergence in bits, 0 <= JS <= 1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return np.nan
    p = p / ps; q = q / qs
    m = (p + q) * 0.5
    def kl(a):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / m[mask]))
    return float(0.5 * (kl(p) + kl(q)))


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _save(fig, folder, stem, show):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(folder / (stem + '.png'), dpi=220, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def _bootstrap_js(a, b, n_boot, rng, batch=100):
    """Bootstrap rows (documents) independently within each source."""
    out = np.empty(n_boot, dtype=float)
    na, nb = len(a), len(b)
    for start in range(0, n_boot, batch):
        end = min(start + batch, n_boot)
        ia = rng.integers(0, na, size=(end-start, na))
        ib = rng.integers(0, nb, size=(end-start, nb))
        sums_a = a[ia].sum(axis=1)
        sums_b = b[ib].sum(axis=1)
        for j in range(end-start):
            out[start+j] = _js(sums_a[j], sums_b[j])
    return out


def analyze_motif_divergence(
    *, motifs_dir, forests_dir, results_dir, genre, n_sentences,
    sources=None, dataset_name=None, prompting=None, attack=None,
    decoding=None, repetition_penalty=None, matched=None, sampling=None,
    tree_set=None, human_source='Human', n_boot=10000,
    confidence=0.95, seed=42, top_k=100, rebuild=False, show=True,
):
    """Return (output_directory, tables) and write CSVs + publication plots.

    Motifs are uniquely identified by canonical structural expression across all
    sources. All motifs are counted against all original RST trees, irrespective
    of which source's grammar originally discovered them.
    """
    if n_boot < 100 or not 0 < confidence < 1:
        raise ValueError('Require n_boot >= 100 and 0 < confidence < 1')
    motifs_dir, forests_dir = Path(motifs_dir), Path(forests_dir)
    results_dir = Path(results_dir)
    allowed = None if sources is None else set(sources)

    # Accept results/motifs/<dataset> and results/motifs (with dataset folders).
    candidates = sorted(motifs_dir.glob('source-*/motifs.csv'))
    if not candidates:
        candidates = sorted(motifs_dir.glob('*/source-*/motifs.csv'))
    if not candidates:
        raise FileNotFoundError(f'No motif CSVs under {motifs_dir}')

    forest_paths = []
    for path in sorted(forests_dir.glob('*.xml')):
        m = _metadata(path)
        if m.get('genre') != str(genre) or m.get('n_sentences') != str(n_sentences):
            continue
        if allowed is not None and m.get('source') not in allowed:
            continue
        forest_paths.append((m['source'], path))
    if not forest_paths:
        raise ValueError('No matching forest XML files')
    selected_sources = sorted(set(s for s, _ in forest_paths))
    if allowed is not None and set(selected_sources) != allowed:
        raise ValueError(f'Missing sources: {allowed - set(selected_sources)}')
    if human_source not in selected_sources:
        raise ValueError(f'Missing human reference: {human_source}')
    if len(forest_paths) != len(selected_sources):
        raise ValueError('Expected exactly one XML forest per source for this dataset')

    # Resolve the exact extraction dataset, avoiding accidental mixing of runs.
    datasets = {p.parent.parent.resolve() for p in candidates}
    if len(datasets) > 1:
        matching = [d for d in datasets if d.name == dataset_name] if dataset_name else []
        if len(matching) != 1:
            raise ValueError('Multiple motif datasets found; pass motifs_dir for ONE dataset')
        candidates = [p for p in candidates if p.parent.parent.resolve() == matching[0]]

    motifs_by_source = defaultdict(set)
    discovered_all = defaultdict(set)
    discovered_wildcard = defaultdict(set)
    motif_structures = {}
    motif_metadata = {}
    for path in candidates:
        source = path.parent.name.removeprefix('source-')
        if source not in selected_sources:
            continue
        frame = pd.read_csv(path)
        if 'motif' not in frame:
            raise ValueError(f'Missing motif column in {path}')
        for record in frame.to_dict('records'):
            tree = _parse_motif(str(record['motif']))
            key = _canonical_text(tree)
            discovered_all[source].add(key)
            if any(node[0] == 'X' for node in _iter_nodes(tree)):
                discovered_wildcard[source].add(key)
                continue
            motifs_by_source[source].add(key)
            motif_structures[key] = tree
            motif_metadata[key] = {
                'has_wildcard': bool('X' in [n[0] for n in _iter_nodes(tree)]),
                'n_relations': sum(n[0] not in ('X', 'EDU') for n in _iter_nodes(tree)),
            }
    if not motif_structures:
        raise ValueError('No non-wildcard motifs in the selected source CSVs')
    for source in selected_sources:
        motifs_by_source.setdefault(source, set())
        discovered_all.setdefault(source, set())
        discovered_wildcard.setdefault(source, set())

    # Globally stable IDs, independent of input order.
    vocabulary = sorted(motif_structures)
    ids = {motif: 'M' + hashlib.sha256(motif.encode()).hexdigest()[:16] for motif in vocabulary}
    if len(set(ids.values())) != len(ids):
        raise ValueError('Motif ID hash collision')
    source_order = [human_source] + [s for s in selected_sources if s != human_source]
    if dataset_name is None:
        dataset_name = '__'.join([
            f'genre-{genre}', f'prompting-{prompting}', f'attack-{attack}',
            f'decoding-{decoding}', f'rep-{repetition_penalty}',
            f'matched-{matched}', f'n_sentences-{n_sentences}',
            f'sampling-{sampling}', f'trees-{tree_set}',
        ])
    out = results_dir / 'divergence_analysis' / _safe(dataset_name)
    figures = out / 'figures'
    out.mkdir(parents=True, exist_ok=True)

    inputs = candidates + [p for _, p in forest_paths]
    settings = dict(version=VERSION, genre=genre, n_sentences=n_sentences,
                    sources=selected_sources, prompting=prompting, attack=attack,
                    decoding=decoding, repetition_penalty=repetition_penalty,
                    matched=matched, sampling=sampling, tree_set=tree_set,
                    n_boot=n_boot, confidence=confidence, seed=seed, top_k=top_k,
                    human_source=human_source,
                    inputs=[(str(p.resolve()), _hash_file(p)) for p in inputs])
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    manifest = out / 'analysis_manifest.json'
    table_files = ['motif_vocabulary', 'motif_discovery_counts', 'motif_document_frequencies',
                   'motif_source_frequencies', 'js_divergence', 'js_bootstrap_draws',
                   'rank_data', 'rank_scatter_data', 'motif_total_counts',
                   'js_human_vs_llm', 'js_llm_vs_llm', 'analysis_settings']
    fig_files = ['01_discovered_motifs', '02_total_motifs',
                 '03_rank_frequency', '04_rank_scatter_grid']
    if not rebuild and manifest.exists() and all((out / (t+'.csv')).exists() for t in table_files) and all((figures / (f+'.png')).exists() and (figures / (f+'.pdf')).exists() for f in fig_files):
        old = json.loads(manifest.read_text())
        if old.get('fingerprint') == fingerprint:
            tables = {t: pd.read_csv(out / (t+'.csv')) for t in table_files}
            print('Loaded divergence analysis cache:', out)
            if show:
                from IPython.display import display, Image
                for f in fig_files:
                    display(Image(filename=str(figures / (f+'.png'))))
            return out, tables

    # Remove obsolete plots from older versions when rebuilding.
    for old_plot in figures.glob('04_rank_scatter_*.png'):
        old_plot.unlink()
    for old_plot in figures.glob('04_rank_scatter_*.pdf'):
        old_plot.unlink()
    for obsolete in ('02_js_divergence', '04_rank_scatter'):
        for extension in ('png', 'pdf'):
            old_plot = figures / f'{obsolete}.{extension}'
            if old_plot.exists():
                old_plot.unlink()

    # Pre-index by root label to avoid checking all motifs at every subtree.
    by_root = defaultdict(list)
    for i, motif in enumerate(vocabulary):
        by_root[motif_structures[motif][0]].append((i, motif_structures[motif]))

    doc_rows = []
    matrices = {}
    for source, forest in forest_paths:
        print(f'Recounting all {len(vocabulary)} motifs in {source}: {forest.name}', flush=True)
        doc_vectors = []
        doc_ids = []
        root = ET.parse(forest).getroot()
        if root.tag != 'forest':
            raise ValueError(f'Unexpected root tag in {forest}')
        for doc in root:
            if not doc.tag.startswith('DOC_') or len(doc) != 1:
                raise ValueError(f'Invalid document wrapper {doc.tag}')
            tree = _rst_node(doc[0])
            vector = np.zeros(len(vocabulary), dtype=np.int64)
            for node in _iter_nodes(tree):
                if node[0] in ('EDU', 'X'):
                    continue
                for i, motif in (by_root.get(node[0], []) + by_root.get('X', [])):
                    if _matches(motif, node):
                        vector[i] += 1
            doc_vectors.append(vector)
            doc_ids.append(doc.tag)
        if not doc_vectors:
            raise ValueError(f'No documents in {forest}')
        matrix = np.stack(doc_vectors)
        matrices[source] = matrix
        for j, doc_id in enumerate(doc_ids):
            for i, freq in enumerate(matrix[j]):
                # Dense table: explicit zero for motifs absent from a document.
                doc_rows.append((source, doc_id, ids[vocabulary[i]], int(freq)))

    doc_freq = pd.DataFrame(doc_rows, columns=['source','document_id','motif_id','frequency'])
    vocab_rows = []
    for motif in vocabulary:
        vocab_rows.append(dict(motif_id=ids[motif], motif=motif,
                               **motif_metadata[motif],
                               discovered_in=';'.join(sorted(s for s in selected_sources if motif in motifs_by_source[s]))))
    vocab_df = pd.DataFrame(vocab_rows)
    discovery = pd.DataFrame([
        dict(source=source, motif_type=kind, n_motifs=count)
        for source in source_order
        for kind, count in (
            ('With wildcard', len(discovered_wildcard[source])),
            ('Without wildcard', len(motifs_by_source[source])),
            ('Total', len(discovered_all[source])),
        )
    ])
    freq_rows = []
    for s in source_order:
        mat = matrices[s]
        totals = mat.sum(axis=0)
        denom = totals.sum()
        for i, motif in enumerate(vocabulary):
            freq_rows.append(dict(source=s, motif_id=ids[motif], frequency=int(totals[i]),
                                  document_frequency=int(np.count_nonzero(mat[:, i])),
                                  n_documents=len(mat), relative_frequency=float(totals[i]/denom) if denom else np.nan))
    source_freq = pd.DataFrame(freq_rows)
    rank_rows = []
    for s in source_order:
        sub = source_freq[source_freq.source == s].sort_values(['frequency', 'motif_id'], ascending=[False, True])
        sub = sub[sub.frequency > 0].copy()
        sub['rank'] = np.arange(1, len(sub)+1)
        rank_rows.append(sub)
    rank_data = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()

    # Every pair, with independent document-level nonparametric bootstrap.
    rng = np.random.default_rng(seed)
    alpha = 1-confidence
    js_rows, js_draw_rows = [], []
    for i, a in enumerate(source_order):
        for b in source_order[i+1:]:
            observed = _js(matrices[a].sum(axis=0), matrices[b].sum(axis=0))
            draws = _bootstrap_js(matrices[a], matrices[b], n_boot, rng)
            valid = draws[np.isfinite(draws)]
            if not len(valid):
                raise ValueError(f'Undefined bootstrap divergence for {a} vs {b}: zero total motif counts')
            lo, hi = np.quantile(valid, [alpha/2, 1-alpha/2])
            js_rows.append(dict(source_a=a, source_b=b, js_divergence_bits=observed,
                                ci_low=lo, ci_high=hi, confidence=confidence,
                                n_boot=n_boot, n_valid=len(valid),
                                n_documents_a=len(matrices[a]), n_documents_b=len(matrices[b])))
            js_draw_rows.extend((a, b, j, float(x)) for j, x in enumerate(draws))
    js_df = pd.DataFrame(js_rows).sort_values('js_divergence_bits', ascending=False).reset_index(drop=True)
    human_js = js_df[(js_df.source_a == human_source) | (js_df.source_b == human_source)].copy()
    human_js['llm_source'] = human_js.apply(
        lambda row: row.source_b if row.source_a == human_source else row.source_a, axis=1)
    human_js = human_js[['llm_source', 'js_divergence_bits', 'ci_low', 'ci_high',
                         'confidence', 'n_boot', 'n_valid', 'n_documents_a', 'n_documents_b']]
    human_js = human_js.sort_values('js_divergence_bits', ascending=False).reset_index(drop=True)
    llm_js = js_df[(js_df.source_a != human_source) & (js_df.source_b != human_source)].copy()
    llm_js = llm_js.sort_values('js_divergence_bits', ascending=False).reset_index(drop=True)
    js_draws = pd.DataFrame(js_draw_rows, columns=['source_a','source_b','iteration','js_divergence_bits'])

    # Single readable headline; details stay in analysis_settings.csv.
    counts = {s: len(matrices[s]) for s in source_order}
    count_text = (f'{next(iter(counts.values()))} samples per source' if len(set(counts.values())) == 1
                  else f'{sum(counts.values())} samples across {len(counts)} sources')
    genre_label = {'reddit': 'Reddit posts', 'news': 'news articles', 'abstracts': 'abstracts'}.get(str(genre).lower(), str(genre))
    title = f'Motif Analysis of {genre_label} with {n_sentences} sentences and {count_text}'
    palette = {s: plt.get_cmap('tab10')(i % 10) for i, s in enumerate(source_order)}

    # 01 — Discovery counts by wildcard status (descriptive; not used for JSD/ranks).
    x = np.arange(len(source_order))
    widths = .38
    without = [len(motifs_by_source[source]) for source in source_order]
    with_wildcard = [len(discovered_wildcard[source]) for source in source_order]
    fig, ax = plt.subplots(figsize=(max(9, len(source_order) * 1.7), 5.5), layout='constrained')
    b1 = ax.bar(x - widths/2, without, widths, label='Without wildcard', color='#467ba4')
    b2 = ax.bar(x + widths/2, with_wildcard, widths, label='With wildcard', color='#dc9b4a')
    ax.bar_label(b1, padding=2, fontsize=9)
    ax.bar_label(b2, padding=2, fontsize=9)
    ax.set_xticks(x, source_order, rotation=20)
    ax.set_ylabel('Distinct motifs discovered by TreeRePair')
    ax.set_ylim(0, max(without + with_wildcard + [1]) * 1.18)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.set_title(title)
    _save(fig, figures, '01_discovered_motifs', show)

    # 02 — Total discovered motifs, with and without wildcards.
    totals = [len(discovered_all[source]) for source in source_order]
    fig, ax = plt.subplots(figsize=(max(9, len(source_order)*1.7), 5.5),
                           layout='constrained')
    bars = ax.bar(source_order, totals, color=[palette[source] for source in source_order])
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_ylabel('Total distinct motifs discovered by TreeRePair')
    ax.set_ylim(0, max(totals) * 1.13 if max(totals) else 1)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.set_title(title)
    _save(fig, figures, '02_total_motifs', show)

    # 03 — Rank-frequency curves, shared panel, positive counts only.
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    for s in source_order:
        sub = rank_data[rank_data.source == s]
        ax.plot(sub['rank'], sub['relative_frequency'], label=s, color=palette[s], lw=1.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Motif frequency rank'); ax.set_ylabel('Relative motif frequency')
    ax.grid(alpha=.2); ax.legend(frameon=False); ax.set_title(title)
    _save(fig, figures, '03_rank_frequency', show)

    # 04 — Compact small-multiples figure, one panel per Human–LLM pair.
    # Each point is the SAME globally identified motif in the two rankings.
    # Only motifs with positive frequencies in both sources have ranks.
    rank_maps = {
        source: dict(zip(
            rank_data.loc[rank_data.source == source, 'motif_id'],
            rank_data.loc[rank_data.source == source, 'rank']
        ))
        for source in source_order
    }
    all_ids = np.array([ids[m] for m in vocabulary])
    human_rank = np.array(
        [rank_maps[human_source].get(m, np.nan) for m in all_ids],
        dtype=float
    )
    other_sources = [s for s in source_order if s != human_source]
    scatter_panels = []
    rank_scatter_rows = []
    max_rank = 1
    for source in other_sources:
        llm_rank = np.array(
            [rank_maps[source].get(m, np.nan) for m in all_ids],
            dtype=float
        )
        shared = np.isfinite(human_rank) & np.isfinite(llm_rank)
        x, y = human_rank[shared], llm_rank[shared]
        if len(x):
            max_rank = max(max_rank, int(np.max(x)), int(np.max(y)))
        scatter_panels.append((source, x, y))
        for motif_id, rank_h, rank_llm in zip(all_ids[shared], x, y):
            rank_scatter_rows.append({
                'source': source, 'motif_id': motif_id,
                'human_rank': int(rank_h), 'source_rank': int(rank_llm)
            })

    ncols = min(3, len(other_sources))
    nrows = int(np.ceil(len(other_sources) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows + 0.55),
        sharex=True, sharey=True, squeeze=False,
        layout='constrained'
    )
    lim = max(2, max_rank * 1.07)
    for ax, (source, x, y) in zip(axes.flat, scatter_panels):
        ax.scatter(x, y, s=12, alpha=.35, color=palette[source],
                   edgecolors='none', rasterized=True)
        ax.plot([1, lim], [1, lim], '--', color='0.45',
                linewidth=1, zorder=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, lim)
        ax.set_ylim(1, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(source, fontsize=10, fontweight='semibold')
        ax.grid(alpha=.16)
        if not len(x):
            ax.text(.5, .5, 'No shared motifs', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9)
    for ax in list(axes.flat)[len(scatter_panels):]:
        ax.set_visible(False)
    fig.supxlabel(f'Motif rank in {human_source}')
    fig.supylabel('Motif rank in LLM')
    fig.suptitle(title, fontsize=12, fontweight='semibold')
    _save(fig, figures, '04_rank_scatter_grid', show)

    rank_scatter_df = pd.DataFrame(
        rank_scatter_rows,
        columns=['source', 'motif_id', 'human_rank', 'source_rank']
    )

    tables = dict(motif_vocabulary=vocab_df, motif_discovery_counts=discovery,
                  motif_document_frequencies=doc_freq, motif_source_frequencies=source_freq,
                  js_divergence=js_df, js_bootstrap_draws=js_draws,
                  rank_data=rank_data, rank_scatter_data=rank_scatter_df,
                  motif_total_counts=pd.DataFrame({'source': source_order, 'n_motifs': totals}),
                  js_human_vs_llm=human_js, js_llm_vs_llm=llm_js,
                  analysis_settings=pd.DataFrame([dict((k, json.dumps(v, default=str) if isinstance(v, (list,dict)) else v)
                                                        for k,v in settings.items() if k != 'inputs')]))
    for name, frame in tables.items():
        frame.to_csv(out / (name+'.csv'), index=False)
    tmp = out / 'analysis_manifest.json.tmp'
    tmp.write_text(json.dumps(dict(fingerprint=fingerprint, settings=settings,
                                   n_motifs=len(vocabulary), n_documents=counts), indent=2, default=str))
    tmp.replace(manifest)
    print(f'Saved {len(vocabulary)} unique motifs and {sum(counts.values())} documents to {out}')
    print('JSD comparisons sorted descending; wildcard motifs excluded from JSD and ranks.')
    print('motif_document_frequencies.csv explicitly includes zero counts.')
    return out, tables


def display_js_tables(results, decimals=4):
    """Display only the two sorted bootstrap JSD comparison tables in a notebook."""
    from IPython.display import display
    human = results['js_human_vs_llm'].sort_values(
        'js_divergence_bits', ascending=False
    ).reset_index(drop=True)
    llm = results['js_llm_vs_llm'].sort_values(
        'js_divergence_bits', ascending=False
    ).reset_index(drop=True)
    print('Jensen–Shannon divergence: Human vs LLM')
    display(human.round(decimals))
    print('Jensen–Shannon divergence: LLM vs LLM')
    display(llm.round(decimals))
