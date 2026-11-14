"""Extract structurally valid RST motifs from TreeRePair textual grammars.

Notebook-facing functions only; no CLI. Terminal arities are recovered
from prefix-tree constraints, then FCNS fragments are decoded into RST trees.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

VERSION = 3
RULE_RE = re.compile(r'^\s*(\d+)\s*->\s*(.*?)\s*$')


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def _source(path):
    for piece in path.stem.split('__'):
        if piece.startswith('source-'):
            return piece[7:]
    return None


def _select(forests_dir, genre, n_sentences, sources):
    folder = Path(forests_dir).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    allowed = set(sources) if sources is not None else None
    selected = []
    for path in sorted(folder.glob('*.xml')):
        parts = path.stem.split('__')
        if genre is not None and f'genre-{genre}' not in parts:
            continue
        if n_sentences is not None and f'n_sentences-{n_sentences}' not in parts:
            continue
        if allowed is not None and _source(path) not in allowed:
            continue
        selected.append(path)
    if not selected:
        raise ValueError('No matching XML forests')
    if allowed is not None and allowed - {_source(p) for p in selected}:
        raise ValueError('Some requested sources are missing')
    return selected


def parse_textual_grammar(path):
    """Parse TreeRePair numeric productions and 1-based terminal names."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    names, ranks, rules = None, {}, {}
    for line in lines:
        s = line.strip()
        if s.startswith('# terminals '):
            count = int(s.split()[-1])
        elif s.startswith('# names '):
            names = s[len('# names '):].split()
        elif s.startswith('# ranks '):
            # TreeRePair uses a compact rank-table representation; it cannot
            # be interpreted as '# ranks <rank> <terminal IDs>'.
            continue
        elif match := RULE_RE.match(s):
            lhs = int(match.group(1))
            rhs = tuple(int(x) for x in match.group(2).split())
            if not rhs or lhs in rules:
                raise ValueError(f'Empty/duplicate production for {lhs}')
            rules[lhs] = rhs
    if names is None or 'count' not in locals():
        raise ValueError('Missing # terminals or # names header')
    if len(names) != count:
        raise ValueError(f'Expected {count} terminal names, found {len(names)}')
    if not rules:
        raise ValueError('No grammar productions found')
    # Terminal IDs are 1..count; count+1 is the formal parameter symbol.
    return names, ranks, rules


def _parse_prefix(tokens, arities, pos=0):
    if pos >= len(tokens):
        raise ValueError('Incomplete prefix production')
    symbol = tokens[pos]
    if symbol not in arities:
        raise ValueError(f'Unknown symbol rank: {symbol}')
    pos += 1
    children = []
    for _ in range(arities[symbol]):
        child, pos = _parse_prefix(tokens, arities, pos)
        children.append(child)
    return (symbol, tuple(children)), pos


def _infer_nonterminal_ranks(names, terminal_ranks, rules):
    """Recover FCNS symbol arities from the grammar's prefix constraints.

    TreeRePair's numeric grammar encodes first-child/next-sibling trees: the
    same XML label may have different ranks. The '# ranks' header is not a
    simple mapping of terminal IDs to ranks; solve the prefix equations instead.
    """
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint

    n = len(names)
    param = n + 1
    nts = set(rules) - {0}
    if not nts or any(x not in nts and x != param and not 1 <= x <= n
                      for rhs in rules.values() for x in rhs):
        raise ValueError('Unknown grammar symbol or missing nonterminal rule')
    known = {param: 0, **{nt: rhs.count(param) for nt, rhs in rules.items() if nt != 0}}
    # Terminal arities for FCNS encoding: EDU may be a leaf or have a sibling;
    # relations/document wrappers have a child and optionally a sibling.
    lower = np.array([0 if name == '<EDU>' else 1 for name in names], dtype=float)
    upper = np.array([1 if name == '<EDU>' else 2 for name in names], dtype=float)
    lower[0] = upper[0] = 1  # the forest container
    constraints = []
    targets_lower = []
    targets_upper = []
    for lhs, rhs in rules.items():
        if lhs == 0:
            # The start rule may be a forest of concatenated trees, and is
            # checked separately after deriving the ranks from NT productions.
            continue
        counts = np.zeros(n, dtype=float)
        known_sum = 0
        for i, symbol in enumerate(rhs):
            if 1 <= symbol <= n:
                counts[symbol - 1] += 1
            else:
                known_sum += known[symbol]
            # A prefix tree cannot close before its final token.
            if i < len(rhs) - 1:
                constraints.append(counts.copy())
                targets_lower.append(i + 1 - known_sum)
                targets_upper.append(np.inf)
        constraints.append(counts.copy())
        targets_lower.append(len(rhs) - 1 - known_sum)
        targets_upper.append(len(rhs) - 1 - known_sum)
    from scipy.sparse import csr_matrix
    matrix = csr_matrix(np.asarray(constraints))
    result = milp(c=np.zeros(n), integrality=np.ones(n),
                  bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, targets_lower, targets_upper),
                  options={'time_limit': 90})
    if result.x is None:
        raise ValueError(f'Could not infer TreeRePair terminal arities: {result.message}')
    ranks = {i + 1: int(round(v)) for i, v in enumerate(result.x)}
    ranks.update(known)
    return ranks, param


def _fcns_to_rst(tree):
    """Decode first-child/next-sibling grammar fragments into RST patterns.

    An unbound parameter denotes one unknown subtree (X). Fragments with
    unresolved sibling chains or incomplete binary relations are discarded.
    """
    def one(t):
        label, children = t
        if label == 'X':
            return ('X', ()), None
        if label == 'EDU':
            if len(children) > 1:
                raise ValueError('EDU has more than one FCNS child')
            sibling = one(children[0])[0] if children else None
            return ('EDU', ()), sibling
        if not label.startswith('R_'):
            raise ValueError('Non-RST wrapper')
        if not children:
            raise ValueError('RST relation without children')
        first, second = one(children[0])
        if second is None:
            raise ValueError('Incomplete binary relation')
        # A relation with two FCNS slots has both a child and a sibling.
        sibling = one(children[1])[0] if len(children) == 2 else None
        if len(children) > 2:
            raise ValueError('Invalid FCNS relation arity')
        return (label, (first, second)), sibling
    node, sibling = one(tree)
    if sibling is not None:
        raise ValueError('Fragment contains a sibling outside its root')
    return node

def _motif_text(tree):
    label, children = tree
    if not children:
        return label
    return label + '(' + ','.join(_motif_text(c) for c in children) + ')'


def _is_motif(tree):
    label, children = tree
    if label in ('EDU', 'X'):
        return not children
    if not label.startswith('R_') or len(children) != 2:
        return False
    return all(_is_motif(child) for child in children)


def _canonical(tree):
    label, children = tree
    if label.startswith('R_'):
        # XML convention: R_<relation>_<nuclearity>, e.g. R_Joint_NN.
        relation, sep, nuclearity = label[2:].rpartition('_')
        if not sep or nuclearity not in ('NN', 'NS', 'SN'):
            raise ValueError(f'Invalid RST relation/nuclearity label: {label}')
        label = nuclearity + '_' + relation.upper()
    return (label, tuple(_canonical(c) for c in children))


def _stats(tree):
    label, children = tree
    if not children:
        return (0, 0, label == 'X')
    sub = [_stats(c) for c in children]
    return (1 + sum(v[0] for v in sub), 1 + max(v[1] for v in sub), any(v[2] for v in sub))


def _rst_subtrees(forest):
    root = ET.parse(forest).getroot()
    if root.tag != 'forest':
        raise ValueError(f'Expected <forest> root in {forest}')
    result = Counter()
    for doc in root:
        if not doc.tag.startswith('DOC_'):
            raise ValueError(f'Unexpected document wrapper {doc.tag}')
        stack = list(doc)
        while stack:
            element = stack.pop()
            if element.tag.startswith('R_'):
                def convert(el):
                    if el.tag == 'EDU':
                        if len(el):
                            raise ValueError('EDU cannot have children')
                        return ('EDU', ())
                    if not el.tag.startswith('R_') or len(el) != 2:
                        raise ValueError(f'Invalid binary RST node {el.tag}, children={len(el)}')
                    return (el.tag, tuple(convert(c) for c in el))
                result[convert(element)] += 1
                stack.extend(list(element))
            elif element.tag != 'EDU':
                raise ValueError(f'Unexpected RST node {element.tag}')
    return result


def _match(pattern, concrete):
    if pattern[0] == 'X':
        return True
    return pattern[0] == concrete[0] and len(pattern[1]) == len(concrete[1]) and all(
        _match(a, b) for a, b in zip(pattern[1], concrete[1])
    )


def _extract_one(grammar, forest, *, min_relations, min_depth, include_wildcards):
    names, terminal_ranks, rules = parse_textual_grammar(grammar)
    arities, parameter = _infer_nonterminal_ranks(names, terminal_ranks, rules)
    # The start production may contain a forest node with variable arity;
    # only numbered NT productions are needed for motif extraction.
    parsed = {}
    for lhs, rhs in rules.items():
        if lhs == 0:
            continue
        node, end = _parse_prefix(rhs, arities)
        if end != len(rhs):
            raise ValueError(f'Production {lhs} has trailing symbols: {rhs[end:]}')
        parsed[lhs] = node

    expanded = {}
    def expand(node, stack=()):
        sid, children = node
        if sid == parameter:
            return ('X', ())
        if 1 <= sid <= len(names):
            return (names[sid - 1].strip("<>"), tuple(expand(c, stack) for c in children))
        if sid in stack:
            raise ValueError(f'Cyclic grammar dependency: {stack + (sid,)}')
        if sid not in parsed:
            raise ValueError(f'Undefined grammar nonterminal {sid}')
        # Replace formal parameters with actual arguments, preserving X for
        # any parameter which remains unbound in a rule template.
        template = expanded.get(sid)
        if template is None:
            template = expand(parsed[sid], stack + (sid,))
            expanded[sid] = template
        arguments = tuple(expand(c, stack) for c in children)
        if len(arguments) != arities[sid]:
            raise ValueError(f'Nonterminal {sid} arity mismatch')
        it = iter(arguments)
        def substitute(t):
            if t[0] == 'X':
                return next(it)
            return (t[0], tuple(substitute(c) for c in t[1]))
        if not arguments:
            return template
        return substitute(template)

    templates = defaultdict(set)
    for sid, node in parsed.items():
        expanded_fcns = expand(node)
        try:
            template = _fcns_to_rst(expanded_fcns)
        except ValueError:
            continue
        if _is_motif(template):
            templates[template].add(sid)

    concrete = _rst_subtrees(forest)
    rows = {}
    for template, origins in templates.items():
        matches = [(tree, freq) for tree, freq in concrete.items() if _match(template, tree)]
        if not matches:
            continue  # A production is not automatically an observed RST subtree.
        candidates = [template]
        # Only grammar-derived templates are motifs. Do not automatically
        # promote arbitrary matched forest subtrees to grammar motifs.
        for candidate in candidates:
            if not _is_motif(candidate):
                continue
            nr, depth, wildcard = _stats(candidate)
            if nr < min_relations or depth < min_depth or (wildcard and not include_wildcards):
                continue
            canonical = _canonical(candidate)
            expression = _motif_text(canonical)
            if candidate == template:
                frequency = sum(freq for _, freq in matches)
            else:
                frequency = concrete[candidate]
            previous = rows.get(expression)
            if previous is None:
                rows[expression] = {
                    'motif': expression,
                    'has_wildcard': wildcard,
                    'n_relations': nr,
                    'depth': depth,
                    'frequency': frequency,
                    'grammar_symbols': sorted(origins),
                }
            else:
                previous['frequency'] = max(previous['frequency'], frequency)
                previous['grammar_symbols'] = sorted(set(previous['grammar_symbols']) | origins)
    return sorted(rows.values(), key=lambda x: (-x['frequency'], x['motif'])), len(templates)


def _dataset_key(forest):
    """Derive dataset identifier from the forest filename, excluding source and run ID."""
    parts = forest.stem.split('__')
    selected = [part for part in parts if not part.startswith('source-')
                and not part.isdigit()]
    if not selected:
        raise ValueError(f'Cannot infer dataset from forest name: {forest.name}')
    return '__'.join(selected)


def _default_results_dir(compression_dir):
    """Find project-level results/ using the data/ ancestor."""
    for parent in (compression_dir, *compression_dir.parents):
        if parent.name == 'data':
            return parent.parent / 'results'
    raise ValueError(
        'Cannot locate project data/ ancestor; pass results_dir=ROOT / "results".'
    )


def extract_motifs(
    *, forests_dir, compression_dir, results_dir=None, genre=None, n_sentences=None,
    sources: Iterable[str] | None = None, rebuild=False,
    min_relations=1, min_depth=1, include_wildcards=True,
    continue_on_error=True,
):
    """Extract motifs into results/motifs/<dataset>/source-<source>/.

    Returns (motifs_dir, per-source results, manifest). When multiple datasets
    are selected, motifs_dir points to results/motifs and each dataset gets its
    own manifest. For a single dataset, motifs_dir points to that dataset.
    """
    if min_relations < 1 or min_depth < 1:
        raise ValueError('min_relations and min_depth must be >= 1')
    forests = _select(forests_dir, genre, n_sentences, sources)
    compression_dir = Path(compression_dir).expanduser().resolve()
    results_dir = (Path(results_dir).expanduser().resolve()
                   if results_dir is not None else _default_results_dir(compression_dir))
    base = results_dir / 'motifs'
    base.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for forest in forests:
        grouped[_dataset_key(forest)].append(forest)

    all_results = []
    manifests = {}
    for dataset, group in sorted(grouped.items()):
        output = base / dataset
        output.mkdir(parents=True, exist_ok=True)
        results = []
        seen_sources = set()
        for i, forest in enumerate(group, 1):
            source = _source(forest)
            if source is None:
                raise ValueError(f'Missing source in forest name: {forest.name}')
            if source in seen_sources:
                raise ValueError(f'Duplicate source {source!r} in dataset {dataset!r}')
            seen_sources.add(source)
            grammar = compression_dir / forest.stem / (forest.stem + '.reptxt')
            dest = output / f'source-{source}'
            dest.mkdir(parents=True, exist_ok=True)
            csv_path, json_path = dest / 'motifs.csv', dest / 'motifs.json'
            meta_path = dest / 'extraction_metadata.json'
            print(f'[{i}/{len(group)}] {dataset} / {source}', flush=True)
            if not grammar.is_file():
                result = {'dataset': dataset, 'source': source,
                          'status': 'MISSING_REPTXT', 'n_motifs': 0,
                          'cached': False, 'output_dir': str(dest)}
                results.append(result)
                if not continue_on_error:
                    raise FileNotFoundError(grammar)
                continue
            signature = {
                'version': VERSION,
                'forest_sha256': _hash(forest),
                'grammar_sha256': _hash(grammar),
                'min_relations': min_relations,
                'min_depth': min_depth,
                'include_wildcards': include_wildcards,
            }
            previous = {}
            if meta_path.is_file():
                try:
                    previous = json.loads(meta_path.read_text(encoding='utf-8'))
                except (ValueError, OSError):
                    pass
            cached = (not rebuild and previous.get('signature') == signature
                      and previous.get('status') == 'COMPLETED'
                      and csv_path.is_file() and json_path.is_file()
                      and previous.get('csv_sha256') == _hash(csv_path)
                      and previous.get('json_sha256') == _hash(json_path))
            if cached:
                result = dict(previous['result'])
                result['cached'] = True
                print(f"  Cached: {result['n_motifs']} motifs", flush=True)
            else:
                try:
                    rows, n_templates = _extract_one(
                        grammar, forest, min_relations=min_relations,
                        min_depth=min_depth, include_wildcards=include_wildcards,
                    )
                    with csv_path.open('w', newline='', encoding='utf-8') as stream:
                        writer = csv.DictWriter(stream, fieldnames=[
                            'motif', 'has_wildcard', 'n_relations', 'depth',
                            'frequency', 'grammar_symbols',
                        ])
                        writer.writeheader()
                        for row in rows:
                            writer.writerow({**row, 'grammar_symbols': json.dumps(row['grammar_symbols'])})
                    _write_json(json_path, rows)
                    result = {
                        'dataset': dataset, 'source': source, 'status': 'COMPLETED',
                        'n_motifs': len(rows),
                        'n_wildcard_motifs': sum(r['has_wildcard'] for r in rows),
                        'n_grammar_templates': n_templates, 'cached': False,
                        'output_dir': str(dest),
                    }
                    _write_json(meta_path, {
                        'signature': signature, 'status': 'COMPLETED',
                        'csv_sha256': _hash(csv_path), 'json_sha256': _hash(json_path),
                        'result': result, 'forest': str(forest), 'grammar': str(grammar),
                        'generated_at': datetime.now(timezone.utc).isoformat(),
                    })
                    print(f"  Extracted: {len(rows)} motifs", flush=True)
                except Exception as exc:
                    result = {
                        'dataset': dataset, 'source': source, 'status': 'FAILED',
                        'n_motifs': 0, 'cached': False, 'error': str(exc),
                        'output_dir': str(dest),
                    }
                    print(f'  ERROR: {exc}', flush=True)
                    if not continue_on_error:
                        raise
            results.append(result)
        manifest = {
            'version': VERSION, 'dataset': dataset,
            'forests_dir': str(Path(forests_dir).resolve()),
            'compression_dir': str(compression_dir),
            'results_dir': str(results_dir), 'results': results,
            'generated_at': datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output / 'motifs_manifest.json', manifest)
        manifests[dataset] = manifest
        all_results.extend(results)
    if len(manifests) == 1:
        dataset = next(iter(manifests))
        return base / dataset, all_results, manifests[dataset]
    return base, all_results, manifests


def display_motifs(motifs_dir, *, show_grammar_symbols=True):
    """Render all motif CSVs in compact, source-separated, non-striped tables.

    Works with either results/motifs/<dataset> or results/motifs.
    Returns a combined pandas DataFrame for downstream analysis.
    """
    import html
    import pandas as pd
    from IPython.display import HTML, display

    root = Path(motifs_dir).expanduser().resolve()
    paths = sorted(root.glob('source-*/motifs.csv'))
    paths += sorted(root.glob('*/source-*/motifs.csv'))
    if not paths:
        raise FileNotFoundError(f'No source-*/motifs.csv found below {root}')
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame.insert(0, 'source', path.parent.name.removeprefix('source-'))
        frame.insert(0, 'dataset', path.parent.parent.name)
        frame['has_wildcard'] = frame['has_wildcard'].astype(str).str.lower().eq('true')
        frames.append(frame)
    all_motifs = pd.concat(frames, ignore_index=True)
    cols = [
        ('motif', 'Motif'), ('frequency', 'Freq.'),
        ('n_relations', 'Rel.'), ('depth', 'Depth'),
        ('has_wildcard', 'Wildcard'),
    ]
    if show_grammar_symbols and 'grammar_symbols' in all_motifs.columns:
        cols.append(('grammar_symbols', 'Grammar symbols'))
    style = """<style>
    .rst-motif-block{margin:14px 0 23px;font-family:Arial,sans-serif}
    .rst-motif-title{font-size:14px;font-weight:700;margin:0 0 5px;
                     padding-bottom:5px;border-bottom:1px solid #98a2b3}
    .rst-motif-count{font-size:11px;font-weight:400;color:#667085;margin-left:12px}
    .rst-motif-table{width:100%;border-collapse:collapse;font-size:11px;
                      line-height:1.2;table-layout:auto}
    .rst-motif-table th{padding:4px 6px;text-align:left;font-weight:600;
                        border-bottom:1px solid #d0d5dd;background:transparent}
    .rst-motif-table td{padding:2px 6px;border-bottom:1px solid #eaecf0;
                        vertical-align:top;background:transparent!important}
    .rst-motif-table .rst-expr{font-family:Consolas,monospace;
                               overflow-wrap:anywhere;min-width:260px}
    .rst-motif-table .rst-num{text-align:right;white-space:nowrap;
                              font-variant-numeric:tabular-nums}
    .rst-motif-table .rst-center{text-align:center;white-space:nowrap}
    </style>"""
    parts = [style]
    for (dataset, source), group in all_motifs.groupby(['dataset', 'source'], sort=True):
        group = group.sort_values(['frequency', 'n_relations', 'motif'],
                                  ascending=[False, False, True])
        with_x = int(group['has_wildcard'].sum())
        heading = (f'{html.escape(dataset)} · {html.escape(source)}'
                   if all_motifs['dataset'].nunique() > 1 else html.escape(source))
        header = ''.join(f'<th>{html.escape(label)}</th>' for _, label in cols)
        rows = []
        for record in group.to_dict('records'):
            cells = []
            for key, _ in cols:
                value = record.get(key, '')
                if pd.isna(value):
                    value = ''
                if key == 'has_wildcard':
                    value = 'Yes' if value else 'No'
                cls = ('rst-expr' if key == 'motif' else
                       'rst-num' if key in ('frequency', 'n_relations', 'depth') else
                       'rst-center' if key == 'has_wildcard' else '')
                cells.append(f'<td class="{cls}">{html.escape(str(value))}</td>')
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        parts.append(
            f'<section class="rst-motif-block">'
            f'<div class="rst-motif-title">{heading}'
            f'<span class="rst-motif-count">{len(group)} motifs · '
            f'{len(group)-with_x} complete · {with_x} wildcard</span></div>'
            f'<table class="rst-motif-table"><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )
    display(HTML(''.join(parts)))
    print(f'{len(all_motifs):,} motifs · {all_motifs["source"].nunique()} sources '
          f'· {all_motifs["dataset"].nunique()} datasets')
    return all_motifs
