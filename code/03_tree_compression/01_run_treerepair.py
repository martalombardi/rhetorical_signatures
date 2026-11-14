"""Run and cache TreeRePair compression of grouped RST XML forests.

Import this module from a notebook. No CLI or import-time side effects.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_name(forest: Path) -> str | None:
    for component in forest.stem.split("__"):
        if component.startswith("source-"):
            return component[len("source-"):]
    return None


def select_forests(
    forests_dir: str | Path,
    *,
    genre: str | None = None,
    n_sentences: int | None = None,
    sources: Iterable[str] | None = None,
) -> list[Path]:
    """Select grouped XML forests by their filename metadata.

    Dataset-level settings (prompting, attack, decoding, etc.) belong to the
    experiment directory and are resolved by the notebook before this call.
    """
    directory = Path(forests_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Forests directory not found: {directory}")
    source_filter = set(sources) if sources is not None else None
    selected = []
    for path in sorted(directory.glob("*.xml")):
        parts = path.stem.split("__")
        if genre is not None and f"genre-{genre}" not in parts:
            continue
        if n_sentences is not None and f"n_sentences-{n_sentences}" not in parts:
            continue
        if source_filter is not None and _source_name(path) not in source_filter:
            continue
        selected.append(path)
    if not selected:
        raise ValueError(
            f"No XML forests match genre={genre!r}, "
            f"n_sentences={n_sentences!r}, sources={source_filter!r} in {directory}"
        )
    if source_filter is not None:
        found = {_source_name(path) for path in selected}
        missing = source_filter - found
        if missing:
            raise ValueError(f"Requested sources not found: {sorted(missing)}")
    return selected


def run_treerepair(
    *,
    forests_dir: str | Path,
    binary: str | Path,
    genre: str | None = None,
    n_sentences: int | None = None,
    sources: Iterable[str] | None = None,
    output_dir: str | Path | None = None,
    rebuild: bool = False,
    textual: bool = True,
    optimize: str = "filesize",
    max_rank: int | None = None,
    timeout: int = 600,
    continue_on_error: bool = True,
) -> tuple[Path, list[dict], dict]:
    """Compress selected forests and reuse matching cached grammars.

    Each forest gets its own directory with .rep, optional .reptxt, logs and
    metadata.json. Cache requires the same XML SHA-256, binary SHA-256,
    compression options, and nonempty expected output files.

    Returns (output_dir, result_records, run_manifest).
    """
    forests_dir = Path(forests_dir)

    if not forests_dir.is_dir():
        raise FileNotFoundError(
            f"Forests directory not found: {forests_dir}"
        )

    if output_dir is None:
        output_dir = forests_dir.parent / "compressed"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
        
    if optimize not in {"filesize", "edges", "disable"}:
        raise ValueError("optimize must be 'filesize', 'edges', or 'disable'")
    if max_rank is not None and max_rank < 0:
        raise ValueError("max_rank must be nonnegative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    forests_dir = Path(forests_dir).expanduser().resolve()
    binary = Path(binary).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"TreeRePair binary not found: {binary}")
    if not binary.stat().st_mode & 0o111:
        raise PermissionError(f"TreeRePair binary is not executable: {binary}")

    forests = select_forests(
        forests_dir, genre=genre, n_sentences=n_sentences, sources=sources
    )
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else forests_dir.parent / "compressed"
    )
    destination.mkdir(parents=True, exist_ok=True)

    options = {"textual": textual, "optimize": optimize, "max_rank": max_rank}
    binary_sha256 = _sha256(binary)
    results: list[dict] = []

    for index, forest in enumerate(forests, 1):
        workdir = destination / forest.stem
        workdir.mkdir(parents=True, exist_ok=True)
        rep = workdir / (forest.stem + ".rep")
        reptxt = workdir / (forest.stem + ".reptxt")
        metadata_path = workdir / "compression_metadata.json"
        source_hash = _sha256(forest)
        expected = [rep] + ([reptxt] if textual else [])
        signature = {
            "forest_sha256": source_hash,
            "binary_sha256": binary_sha256,
            "options": options,
        }
        previous = {}
        if metadata_path.is_file():
            try:
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                previous = {}

        cached = (
            not rebuild
            and previous.get("signature") == signature
            and previous.get("returncode") == 0
            and all(p.is_file() and p.stat().st_size > 0 for p in expected)
        )
        print(f"[{index}/{len(forests)}] {_source_name(forest) or forest.stem}", flush=True)
        if cached:
            status, returncode = "CACHED_UNVALIDATED", 0
            print("  Cached grammar reused", flush=True)
        else:
            # Do not overwrite an old grammar before successful recompression.
            # TreeRePair determines output basenames from the input filename.
            staging = workdir / "_staging"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            local_xml = staging / forest.name
            shutil.copy2(forest, local_xml)
            command = [str(binary), "-optimize", optimize]
            if textual:
                command.append("-textual")
            if max_rank is not None:
                command += ["-max_rank", str(max_rank)]
            command.append(str(local_xml))
            try:
                proc = subprocess.run(
                    command,
                    cwd=staging,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
                returncode = proc.returncode
                stdout, stderr = proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = None
                stdout = exc.stdout or b""
                stderr = exc.stderr or b""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors="replace")
                stderr += "\nTreeRePair timed out."
            (workdir / "stdout.log").write_text(stdout, encoding="utf-8")
            (workdir / "stderr.log").write_text(stderr, encoding="utf-8")
            generated_rep = staging / rep.name
            generated_text = staging / reptxt.name
            has_rep = generated_rep.is_file() and generated_rep.stat().st_size > 0
            has_text = generated_text.is_file() and generated_text.stat().st_size > 0
            if returncode == 0 and has_rep and (has_text or not textual):
                shutil.move(str(generated_rep), str(rep))
                if textual:
                    shutil.move(str(generated_text), str(reptxt))
                else:
                    reptxt.unlink(missing_ok=True)
                status = "COMPLETED_UNVALIDATED"
            elif returncode is None:
                status = "TIMEOUT"
            elif returncode == -11 and has_rep:
                status = "CRASHED_WITH_OUTPUT_NOT_CACHED"
            else:
                status = "FAILED"
            shutil.rmtree(staging)
            metadata = {
                "signature": signature,
                "forest": str(forest),
                "binary": str(binary),
                "command": command,
                "status": status,
                "returncode": returncode,
                "updated_utc": _utc_now(),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  {status} (returncode={returncode})", flush=True)

        result = {
            "source": _source_name(forest),
            "forest": str(forest),
            "status": status,
            "returncode": returncode,
            "cached": cached,
            "rep_file": str(rep) if rep.is_file() else None,
            "reptxt_file": str(reptxt) if textual and reptxt.is_file() else None,
            "rep_bytes": rep.stat().st_size if rep.is_file() else 0,
            "reptxt_bytes": reptxt.stat().st_size if reptxt.is_file() else 0,
            "forest_sha256": source_hash,
        }
        results.append(result)
        if status in {"FAILED", "TIMEOUT", "CRASHED_WITH_OUTPUT_NOT_CACHED"} and not continue_on_error:
            raise RuntimeError(f"TreeRePair failed for {forest}: {status}")

    manifest = {
        "created_utc": _utc_now(),
        "forests_dir": str(forests_dir),
        "output_dir": str(destination),
        "selection": {"genre": genre, "n_sentences": n_sentences, "sources": list(sources) if sources is not None else None},
        "options": options,
        "results": results,
    }
    (destination / "compression_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination, results, manifest
