"""Editable experiment parameters. Project-wide constants live in config.py."""

# RAID filters
DOMAIN = "reddit"
PROMPTING = None
ATTACK = "null"
DECODING = "greedy"
REPETITION_PENALTY = "no"

# Sentence analysis (shared by matched and unmatched)
SENTENCE_ANALYSIS_REBUILD = False
SENTENCE_ANALYSIS_SHOW_PROGRESS = True

# Dataset selection
SELECTION_MODE = "matched"  # "matched" or "unmatched"
K_SENTENCES = 6                # Matched: minimum; unmatched: first k retained
M_DOCUMENTS = None             # None: all matched groups / max balanced unmatched
N_BINS = 10                    # Unmatched length-stratification quantiles
SEED = 42                      # Explicit reproducible sampling seed
SELECTION_SAVE = True
SELECTION_PLOT = False  # Generate the two figures explicitly in the notebook
# Matched always requires Human and all configured LLMs.
# Unmatched also includes all configured sources in the current implementation.

# RST parsing
# For unmatched, set equal to K_SENTENCES; for matched, downstream parsing
# must be checked to ensure full texts are used rather than truncated.
N_SENTENCES = 6
RST_MODEL = "tchewik/isanlp_rst_v3"
RST_MODEL_VERSION = "rstdt"
RST_DEVICE = "auto"
RST_LIMIT = 200
RST_REBUILD = False
RST_RETRY_ERRORS = False

# XML forests
FOREST_GROUP_BY = ["genre", "source", "n_sentences"]
FOREST_REBUILD = False
FOREST_VALIDATE = True
FOREST_METADATA_FILE = None

# TreeRePair compression
COMPRESSION_GENRE = None
COMPRESSION_N_SENTENCES = None
COMPRESSION_SOURCES = None
COMPRESSION_REBUILD = False
COMPRESSION_TEXTUAL = True
COMPRESSION_OPTIMIZE = "filesize"
COMPRESSION_MAX_RANK = None
COMPRESSION_TIMEOUT = 600
COMPRESSION_CONTINUE_ON_ERROR = True

# TreeDePair validation
VALIDATION_REBUILD = False
VALIDATION_TIMEOUT = 600
VALIDATION_KEEP_RECONSTRUCTED = True

# RST motif extraction
MOTIF_RESULTS_DIR = None
MOTIF_REBUILD = False
MOTIF_MIN_RELATIONS = 1
MOTIF_MIN_DEPTH = 1
MOTIF_INCLUDE_WILDCARDS = True
MOTIF_CONTINUE_ON_ERROR = True
MOTIF_SHOW_GRAMMAR_SYMBOLS = True

# RST structural analysis
STRUCTURAL_REBUILD = False
STRUCTURAL_SOURCES = None
STRUCTURAL_N_BOOT = 10000
STRUCTURAL_CONFIDENCE = 0.95
STRUCTURAL_SEED = 42
STRUCTURAL_HUMAN_SOURCE = "Human"
STRUCTURAL_DELTA_METHOD = "independent"  # Set 'paired' for matched, if supported
STRUCTURAL_HEX_GRIDSIZE = 14
STRUCTURAL_SHOW = True
STRUCTURAL_TREE_SET = "first200_per_source"
