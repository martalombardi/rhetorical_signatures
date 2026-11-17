
# ============================================================
# 01 — Environment, imports and experiment configuration
# ============================================================

import sys
import subprocess
import importlib
import importlib.util
import json
from pathlib import Path

# ------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------





# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import torch

from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
from IPython.display import display

from isanlp.annotation_rst import DiscourseUnit
from isanlp_rst.parser import Parser

# ------------------------------------------------------------
# spaCy English model
# ------------------------------------------------------------

if not spacy.util.is_package("en_core_web_sm"):
    subprocess.check_call([
        sys.executable, "-m", "spacy",
        "download", "en_core_web_sm"
    ])

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CODE_DIR = ROOT / "code"

PROCESSING_DIR = CODE_DIR / "01_data_processing"
RST_DIR = CODE_DIR / "02_rst_parsing"
COMPRESSION_DIR = CODE_DIR / "03_tree_compression"

RST_DATA_DIR = ROOT / "data" / "rst"

for directory in (
    PROCESSING_DIR,
    RST_DIR,
    COMPRESSION_DIR,
):
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

RST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------

import config
importlib.reload(config)

from config import (
    TRAIN_FILE,
    RESULTS_DIR,
    FILTERED_DIR,
    MATCHED_DIR,
    UNMATCHED_DIR,
    RANDOM_SEED,
    MODEL_ORDER,
    MODEL_NAMES,
)

# ------------------------------------------------------------
# Reusable module loader
# ------------------------------------------------------------

def load_project_module(name, script_path):
    script_path = Path(script_path)

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        name, script_path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module

# ------------------------------------------------------------
# Load dataset selection modules
# ------------------------------------------------------------

unmatched_selector = load_project_module(
    "unmatched_selector",
    PROCESSING_DIR / "02_unmatched_dataset_selection.py",
)
matched_selector = load_project_module(
    "matched_selector",
    PROCESSING_DIR / "03_matched_dataset_selection.py",
)

# Keep the shared sentence-analysis API used by existing notebook cells.
process_sentences = unmatched_selector
SELECTION_VERSION = {
    "matched": matched_selector.SELECTION_VERSION,
    "unmatched": unmatched_selector.SELECTION_VERSION,
}

# ------------------------------------------------------------
# Load RST parsing script
# ------------------------------------------------------------

rst = load_project_module(
    "rst_parsing",
    RST_DIR / "01_parse_documents.py",
)

# ------------------------------------------------------------
# Load XML forest construction script
# ------------------------------------------------------------

forest_builder = load_project_module(
    "xml_forest_builder",
    RST_DIR / "02_build_xml_forests.py",
)

# ------------------------------------------------------------
# Load TreeRePair compression script
# ------------------------------------------------------------

COMPRESSION_SCRIPT = (
    COMPRESSION_DIR / "01_run_treerepair.py"
)

tree_compressor = load_project_module(
    "tree_compressor",
    COMPRESSION_SCRIPT,
)

# ------------------------------------------------------------
# Load TreeRePair validation script
# ------------------------------------------------------------

VALIDATION_SCRIPT = (
    COMPRESSION_DIR / "02_validate_grammar.py"
)

grammar_validator = load_project_module(
    "grammar_validator",
    VALIDATION_SCRIPT,
)

# ------------------------------------------------------------
# Load RST motif extraction script
# ------------------------------------------------------------

MOTIF_SCRIPT = (
    COMPRESSION_DIR / "03_extract_motifs.py"
)

motif_extractor = load_project_module(
    "motif_extractor",
    MOTIF_SCRIPT,
)


# ------------------------------------------------------------
# Load RST structural analysis module
# ------------------------------------------------------------

STRUCTURAL_DIR = CODE_DIR / "04_structural_analysis"

STRUCTURAL_SCRIPT = (
    STRUCTURAL_DIR / "01_inspect_rst_trees.py"
)

structural_inspector = load_project_module(
    "structural_inspector",
    STRUCTURAL_SCRIPT,
)


# ------------------------------------------------------------
# Motif divergence analysis
# ------------------------------------------------------------

DIVERGENCE_SCRIPT = (
    CODE_DIR
    / "05_divergence_analysis"
    / "01_analyze_motif_divergence.py"
)

divergence_analyzer = load_project_module(
    "divergence_analyzer",
    DIVERGENCE_SCRIPT,
)


DIVERSITY_SCRIPT = (
    CODE_DIR
    / "05_divergence_analysis"
    / "02_analyze_motif_diversity.py"
)

diversity_analyzer = load_project_module(
    "diversity_analyzer",
    DIVERSITY_SCRIPT,
)

# ------------------------------------------------------------
# TreeRePair binaries
# ------------------------------------------------------------

TREEREPAIR_VENDOR_DIR = (
    CODE_DIR
    / "03_tree_compression" / "treerepair"
    / "vendor"
    / "TreeRePair"
)

TREEREPAIR_BINARY = (
    TREEREPAIR_VENDOR_DIR
    / "TreeRePair"
    / "TreeRePair"
)

TREEDEPAIR_BINARY = (
    TREEREPAIR_VENDOR_DIR
    / "TreeDePair"
    / "TreeDePair"
)

# ------------------------------------------------------------
# Load experiment parameters (edit experiment_config.py only)
# ------------------------------------------------------------

EXPERIMENT_CONFIG_FILE = CODE_DIR / "experiment_config.py"
experiment_config = load_project_module(
    "experiment_config", EXPERIMENT_CONFIG_FILE
)

# Preserve the original notebook's global variable names for cells 02+.
# Export only uppercase configuration constants.
for _name, _value in vars(experiment_config).items():
    if _name.isupper() and not _name.startswith("_"):
        globals()[_name] = _value

if SEED is None:
    SEED = RANDOM_SEED
if MOTIF_RESULTS_DIR is None:
    MOTIF_RESULTS_DIR = RESULTS_DIR / "motifs"

if SELECTION_MODE not in {"matched", "unmatched"}:
    raise ValueError("SELECTION_MODE must be 'matched' or 'unmatched'")
if isinstance(K_SENTENCES, bool) or not isinstance(K_SENTENCES, int) or K_SENTENCES < 1:
    raise ValueError("K_SENTENCES must be a positive integer")
if M_DOCUMENTS is not None and (
    isinstance(M_DOCUMENTS, bool)
    or not isinstance(M_DOCUMENTS, int)
    or M_DOCUMENTS < 1
):
    raise ValueError("M_DOCUMENTS must be None or a positive integer")
if not isinstance(N_BINS, int) or isinstance(N_BINS, bool) or N_BINS < 1:
    raise ValueError("N_BINS must be a positive integer")
if not isinstance(SEED, int) or isinstance(SEED, bool):
    raise ValueError("SEED must be an integer or None in experiment_config.py")

# Compatibility with existing RST notebook cells that reference RST_MATCHED.
RST_MATCHED = SELECTION_MODE == "matched"
dataset_selector = matched_selector if RST_MATCHED else unmatched_selector

if not RST_MATCHED and N_SENTENCES != K_SENTENCES:
    raise ValueError("For unmatched selection, N_SENTENCES must equal K_SENTENCES")

# Both selection modules accept output_root; it is the parent of matched/unmatched.
SELECTION_OUTPUT_ROOT = FILTERED_DIR

# ------------------------------------------------------------
# Resolve compression selection
# ------------------------------------------------------------

COMPRESSION_GENRE = (
    DOMAIN
    if COMPRESSION_GENRE is None
    else COMPRESSION_GENRE
)

COMPRESSION_N_SENTENCES = (
    N_SENTENCES
    if COMPRESSION_N_SENTENCES is None
    else COMPRESSION_N_SENTENCES
)

# ------------------------------------------------------------
# Run identification
# ------------------------------------------------------------

analysis_meta = {
    "domain": process_sentences._domain_key(DOMAIN),
    "prompting": PROMPTING,
    "attack": ATTACK,
    "decoding": DECODING,
    "repetition_penalty": REPETITION_PENALTY,
}

run_name = process_sentences._run_name(
    analysis_meta
)

analysis_dir = RESULTS_DIR / run_name
selection_base = MATCHED_DIR if RST_MATCHED else UNMATCHED_DIR

# ------------------------------------------------------------
# Plot configuration
# ------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (9, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------
# Environment summary
# ------------------------------------------------------------

print("\nEnvironment ready")
print("-" * 55)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"spaCy: {spacy.__version__}")

print("\nDataset selection")
print("-" * 55)

print(f"Selection algorithm versions: {SELECTION_VERSION}")
print(f"Mode: {SELECTION_MODE}")
print(f"k sentences: {K_SENTENCES}")
print(f"m documents per source: {M_DOCUMENTS}")
print(f"Unmatched length bins: {N_BINS if not RST_MATCHED else 'N/A'}")
print(f"Seed: {SEED}")
print(f"Save selection: {SELECTION_SAVE}")
print(f"Generate selection plots: {SELECTION_PLOT}")

print("\nRST parsing")
print("-" * 55)

print(f"Number of sentences: {N_SENTENCES}")
print(f"Matched: {RST_MATCHED}")
print(f"Model: {RST_MODEL}")
print(f"Model version: {RST_MODEL_VERSION}")
print(f"Device: {RST_DEVICE}")
print(f"Document limit: {RST_LIMIT}")
print(f"Retry errors: {RST_RETRY_ERRORS}")

print("\nXML forests")
print("-" * 55)

print(f"Grouping variables: {FOREST_GROUP_BY}")
print(f"Rebuild: {FOREST_REBUILD}")
print(f"Validate: {FOREST_VALIDATE}")
print(f"External metadata: {FOREST_METADATA_FILE}")

print("\nTreeRePair compression")
print("-" * 55)

print(f"Genre: {COMPRESSION_GENRE}")
print(f"Sentences: {COMPRESSION_N_SENTENCES}")
print(f"Sources: {COMPRESSION_SOURCES}")
print(f"Rebuild: {COMPRESSION_REBUILD}")
print(f"Textual grammar: {COMPRESSION_TEXTUAL}")
print(f"Optimization: {COMPRESSION_OPTIMIZE}")
print(f"Max rank: {COMPRESSION_MAX_RANK}")
print(f"Timeout: {COMPRESSION_TIMEOUT}s")

print("\nTreeDePair validation")
print("-" * 55)

print(f"Rebuild: {VALIDATION_REBUILD}")
print(f"Timeout: {VALIDATION_TIMEOUT}s")
print(
    "Keep reconstructed XML:",
    VALIDATION_KEEP_RECONSTRUCTED,
)

print("\nRST motif extraction")
print("-" * 55)

print(f"Results directory: {MOTIF_RESULTS_DIR}")
print(f"Rebuild: {MOTIF_REBUILD}")
print(f"Minimum relations: {MOTIF_MIN_RELATIONS}")
print(f"Minimum depth: {MOTIF_MIN_DEPTH}")
print(f"Include wildcards: {MOTIF_INCLUDE_WILDCARDS}")
print(f"Continue on error: {MOTIF_CONTINUE_ON_ERROR}")
print(f"Show grammar symbols: {MOTIF_SHOW_GRAMMAR_SYMBOLS}")

print("\nProject paths")
print("-" * 55)

print(f"Experiment config: {EXPERIMENT_CONFIG_FILE}")
print(f"Domain: {DOMAIN} | Prompting: {PROMPTING} | Attack: {ATTACK}")
print(f"Decoding: {DECODING} | Repetition penalty: {REPETITION_PENALTY}")
print(f"RAID dataset: {TRAIN_FILE}")
print(f"Sentence analysis: {analysis_dir}")
print(f"Selected datasets root: {selection_base}")
print(f"RST output root: {RST_DATA_DIR}")
print(f"Forest builder: {RST_DIR / '02_build_xml_forests.py'}")
print(f"Compression script: {COMPRESSION_SCRIPT}")
print(f"Validation script: {VALIDATION_SCRIPT}")
print(f"Motif extraction script: {MOTIF_SCRIPT}")
print(f"Motif results: {MOTIF_RESULTS_DIR}")
print(f"TreeRePair binary: {TREEREPAIR_BINARY}")
print(f"TreeDePair binary: {TREEDEPAIR_BINARY}")


print("\nRST structural analysis")
print("-" * 55)
print(f"Bootstrap iterations: {STRUCTURAL_N_BOOT}")
print(f"Confidence level: {STRUCTURAL_CONFIDENCE}")
print(f"Delta method: {STRUCTURAL_DELTA_METHOD}")
print(f"Rebuild: {STRUCTURAL_REBUILD}")
print(f"Module: {STRUCTURAL_SCRIPT}")