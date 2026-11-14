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
