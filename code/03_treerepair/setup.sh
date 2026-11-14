#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/vendor/TreeRePair"
if [[ ! -f "$SRC/Makefile.static" ]]; then
  echo "Missing TreeRePair sources: $SRC" >&2
  exit 1
fi
for tool in make g++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool; install build-essential first" >&2
    exit 1
  fi
done
if [[ "${1:-}" == '--clean' ]]; then
  make -C "$SRC" -f Makefile.static clean
fi
if [[ ! -x "$SRC/TreeRePair/TreeRePair" || ! -x "$SRC/TreeDePair/TreeDePair" ]]; then
  make -C "$SRC" -f Makefile.static -j1
fi
test -x "$SRC/TreeRePair/TreeRePair"
test -x "$SRC/TreeDePair/TreeDePair"
echo "TreeRePair: $SRC/TreeRePair/TreeRePair"
echo "TreeDePair: $SRC/TreeDePair/TreeDePair"
