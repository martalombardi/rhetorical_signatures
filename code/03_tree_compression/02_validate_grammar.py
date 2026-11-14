"""Validate TreeRePair grammars by TreeDePair round-trip reconstruction.

Notebook-importable functions only; no CLI and no import-time side effects.
The original XML forest is compared with the reconstructed XML as an ordered,
labelled tree. Validation does not rerun TreeRePair or modify .rep files.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

VALIDATION_SCHEMA = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_name(forest: Path) -> str | None:
    for part in forest.stem.split('__'):
        if part.startswith('source-'):
            return part[len('source-'):]
    return None


def _tree_signature(root: ET.Element) -> tuple:
    """Iterative, ordered XML signature; ignores formatting-only whitespace.

    Attributes and non-whitespace element text/tails are retained. XML comments
    are not included by ElementTree's default parser.
    """
    results = {}
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(list(node)))
        else:
            results[id(node)] = (
                node.tag,
                tuple(sorted(node.attrib.items())),
                (node.text or '').strip(),
                (node.tail or '').strip(),
                tuple(results.pop(id(child)) for child in node),
            )
    return results[id(root)]


def compare_xml_forests(original: str | Path, reconstructed: str | Path) -> dict:
    """Check ordered, labelled structural equality and document counts."""
    original = Path(original)
    reconstructed = Path(reconstructed)
    a = ET.parse(original).getroot()
    b = ET.parse(reconstructed).getroot()
    a_docs = [child.tag for child in a]
    b_docs = [child.tag for child in b]
    same_root = a.tag == b.tag
    same_docs = a_docs == b_docs
    same_structure = _tree_signature(a) == _tree_signature(b)
    return {
        'original_root': a.tag,
        'reconstructed_root': b.tag,
        'original_trees': len(a_docs),
        'reconstructed_trees': len(b_docs),
        'document_ids_match': same_docs,
        'structure_match': same_structure,
        'validated': same_root and same_docs and same_structure,
    }


def validate_grammars(
    *,
    forests_dir: str | Path,
    compression_dir: str | Path,
    binary: str | Path,
    genre: str | None = None,
    n_sentences: int | None = None,
    sources: Iterable[str] | None = None,
    rebuild: bool = False,
    timeout: int = 600,
    continue_on_error: bool = True,
    keep_reconstructed: bool = True,
) -> tuple[Path, list[dict], dict]:
    """Validate selected compressed forests and cache successful comparisons.

    Expects the directory layout produced by 01_run_treerepair.py.
    Returns (validation_dir, result_records, validation_manifest).
    """
    if timeout <= 0:
        raise ValueError('timeout must be positive')
    forests_dir = Path(forests_dir).expanduser().resolve()
    compression_dir = Path(compression_dir).expanduser().resolve()
    binary = Path(binary).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f'TreeDePair executable not found: {binary}')
    if not binary.stat().st_mode & 0o111:
        raise PermissionError(f'TreeDePair executable is not executable: {binary}')
    if not compression_dir.is_dir():
        raise FileNotFoundError(f'Compression directory not found: {compression_dir}')

    # Reuse the exact selection logic from the compression module if present;
    # selection is deliberately implemented here as well to keep modules independent.
    source_filter = set(sources) if sources is not None else None
    forests = []
    for path in sorted(forests_dir.glob('*.xml')):
        parts = path.stem.split('__')
        if genre is not None and f'genre-{genre}' not in parts:
            continue
        if n_sentences is not None and f'n_sentences-{n_sentences}' not in parts:
            continue
        if source_filter is not None and _source_name(path) not in source_filter:
            continue
        forests.append(path.resolve())
    if not forests:
        raise ValueError('No matching XML forests found')
    if source_filter is not None:
        missing = source_filter - {_source_name(p) for p in forests}
        if missing:
            raise ValueError(f'Requested sources not found: {sorted(missing)}')

    validation_dir = compression_dir / 'validation'
    validation_dir.mkdir(parents=True, exist_ok=True)
    decoder_hash = _sha256(binary)
    results = []

    for index, forest in enumerate(forests, 1):
        source = _source_name(forest)
        rep = compression_dir / forest.stem / (forest.stem + '.rep')
        workdir = validation_dir / forest.stem
        workdir.mkdir(parents=True, exist_ok=True)
        metadata_file = workdir / 'validation_metadata.json'
        restored_file = workdir / (forest.stem + '.rep_uncompressed.xml')
        print(f'[{index}/{len(forests)}] {source or forest.stem}', flush=True)

        base = {
            'source': source,
            'forest': str(forest),
            'rep_file': str(rep),
            'restored_file': str(restored_file) if keep_reconstructed else None,
            'cached': False,
            'returncode': None,
            'original_trees': None,
            'reconstructed_trees': None,
            'document_ids_match': None,
            'structure_match': None,
        }
        if not rep.is_file() or rep.stat().st_size == 0:
            result = {**base, 'status': 'MISSING_REP'}
            results.append(result)
            print('  MISSING_REP', flush=True)
            if not continue_on_error:
                raise FileNotFoundError(rep)
            continue

        signature = {
            'schema': VALIDATION_SCHEMA,
            'forest_sha256': _sha256(forest),
            'rep_sha256': _sha256(rep),
            'treedepair_sha256': decoder_hash,
        }
        previous = {}
        if metadata_file.is_file():
            try:
                previous = json.loads(metadata_file.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                pass
        can_cache = (
            not rebuild
            and previous.get('signature') == signature
            and previous.get('status') == 'VALIDATED'
            and (
                not keep_reconstructed
                or (
                    restored_file.is_file()
                    and previous.get('restored_sha256') == _sha256(restored_file)
                )
            )
        )
        if can_cache:
            result = {**base, **previous['result'], 'cached': True}
            results.append(result)
            print('  VALIDATED (cached)', flush=True)
            continue

        restored_file.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix='treedepair_', dir=workdir) as temp:
            staging = Path(temp)
            local_rep = staging / rep.name
            shutil.copy2(rep, local_rep)
            command = [str(binary), str(local_rep)]
            try:
                proc = subprocess.run(
                    command, cwd=staging, capture_output=True,
                    text=True, errors='replace', timeout=timeout, check=False,
                )
                returncode = proc.returncode
                stdout, stderr = proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = None
                stdout = exc.stdout or b''
                stderr = exc.stderr or b''
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors='replace')
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors='replace')
                stderr += '\nTreeDePair timed out.'
            (workdir / 'stdout.log').write_text(stdout, encoding='utf-8')
            (workdir / 'stderr.log').write_text(stderr, encoding='utf-8')
            base['returncode'] = returncode

            # TreeDePair convention: <input>.rep_uncompressed.xml.
            # Search staging and its parent as a fallback, without guessing
            # that a missing output means a structurally invalid grammar.
            candidates = sorted(staging.rglob('*.xml'))
            expected = staging / (local_rep.name + '_uncompressed.xml')
            if returncode is None:
                status = 'TIMEOUT'
            elif returncode != 0:
                status = 'DECOMPRESSION_FAILED'
            elif expected.is_file():
                chosen = expected
                status = None
            elif len(candidates) == 1:
                chosen = candidates[0]
                status = None
            else:
                status = 'OUTPUT_NOT_IDENTIFIED'

            if status is None:
                try:
                    comparison = compare_xml_forests(forest, chosen)
                    base.update(comparison)
                    status = 'VALIDATED' if comparison['validated'] else 'STRUCTURE_MISMATCH'
                    if keep_reconstructed:
                        shutil.copy2(chosen, restored_file)
                except (ET.ParseError, ValueError, RecursionError) as exc:
                    status = 'INVALID_XML'
                    base['error'] = str(exc)
            result = {**base, 'status': status}

        metadata = {
            'signature': signature,
            'status': result['status'],
            'result': result,
            'restored_sha256': (
                _sha256(restored_file)
                if keep_reconstructed and restored_file.is_file()
                else None
            ),
            'updated_utc': _utc_now(),
        }
        metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        results.append(result)
        print(f"  {result['status']} (returncode={result['returncode']})", flush=True)
        if result['status'] != 'VALIDATED' and not continue_on_error:
            raise RuntimeError(f"Validation failed for {forest.name}: {result['status']}")

    manifest = {
        'created_utc': _utc_now(),
        'forests_dir': str(forests_dir),
        'compression_dir': str(compression_dir),
        'validation_dir': str(validation_dir),
        'treedepair_binary': str(binary),
        'selection': {
            'genre': genre,
            'n_sentences': n_sentences,
            'sources': sorted(source_filter) if source_filter is not None else None,
        },
        'results': results,
    }
    (validation_dir / 'validation_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return validation_dir, results, manifest
