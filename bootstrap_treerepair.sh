#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
DEST="$ROOT/code/02_treerepair"
VENDOR="$DEST/vendor/TreeRePair"
URL='https://github.com/dc0d32/TreeRePair.git'

mkdir -p "$DEST/vendor"
if [[ -e "$VENDOR" ]]; then
  echo "Already exists: $VENDOR" >&2
  echo 'Refusing to overwrite existing sources.' >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone "$URL" "$TMP/TreeRePair"
REV="$(git -C "$TMP/TreeRePair" rev-parse HEAD)"
# Vendor plain source files, without the nested Git repository.
rm -rf "$TMP/TreeRePair/.git"
mv "$TMP/TreeRePair" "$VENDOR"
printf '%s\n' "$REV" > "$DEST/UPSTREAM_COMMIT"

cat > "$DEST/setup.sh" <<'SETUP'
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
SETUP
chmod +x "$DEST/setup.sh"

cat > "$DEST/.gitignore" <<'IGNORE'
# Build products
*.o
*.a
*.so
*.d
TreeRePair
TreeDePair
# These patterns are relative to vendor subdirectories; unignore source directories
!vendor/TreeRePair/TreeRePair/
!vendor/TreeRePair/TreeDePair/
# Runtime data
*.rep
*.reptxt
*.log
__pycache__/
IGNORE
# Avoid directory-name ignore collisions; restrict executable ignores to exact paths.
cat > "$DEST/.gitignore" <<'IGNORE'
**/*.o
**/*.a
**/*.so
**/*.d
/vendor/TreeRePair/TreeRePair/TreeRePair
/vendor/TreeRePair/TreeDePair/TreeDePair
**/*.rep
**/*.reptxt
**/*.log
**/__pycache__/
IGNORE

cat > "$DEST/README.md" <<'README'
# TreeRePair integration

Original source: https://github.com/dc0d32/TreeRePair
Pinned upstream revision: see UPSTREAM_COMMIT.
Original GPL-3.0 license is retained in vendor/TreeRePair.

Build in a fresh Colab runtime:

    bash code/02_treerepair/setup.sh

For a full rebuild:

    bash code/02_treerepair/setup.sh --clean

The vendored C++ code is currently UNPATCHED. The memory-cleanup fix must be
recovered and validated separately before production RST forest compression.
README

echo "Vendored TreeRePair at revision $REV"
echo "Next: bash code/02_treerepair/setup.sh"
echo 'Review git status and commit the new source files.'
