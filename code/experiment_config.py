"""Editable parameters for the RST pipeline.

Edit this file to run a new experiment. The existing code/config.py remains
the source of project-wide paths and model constants.
"""

# ------------------------------------------------------------
# RAID experiment filters
# ------------------------------------------------------------

DOMAIN = "reddit"
PROMPTING = None
ATTACK = "null"
DECODING = "greedy"
REPETITION_PENALTY = "no"

# ------------------------------------------------------------
# Sentence analysis
# ------------------------------------------------------------

# Force recomputation even if cached results exist
SENTENCE_ANALYSIS_REBUILD = False

# Show spaCy parsing progress
SENTENCE_ANALYSIS_SHOW_PROGRESS = True

# ------------------------------------------------------------
# Dataset selection
# ------------------------------------------------------------

MIN_SENTENCES = 10
MAX_SENTENCES = 10

# Preserve each source's original length distribution.
# Target = minimum eligible document count across LLMs.
SAMPLING = "proportional"

N_BINS = 10
SEED = None  # None = use RANDOM_SEED from the existing config.py
INCLUDE_HUMAN = True

# ------------------------------------------------------------
# RST parsing configuration
# ------------------------------------------------------------

# Exact number of sentences per document to parse.
# Independent of the preprocessing selection range.
N_SENTENCES = 10

# Which selected dataset to parse
RST_MATCHED = False

# Parser configuration
RST_MODEL = "tchewik/isanlp_rst_v3"
RST_MODEL_VERSION = "rstdt"
RST_DEVICE = "auto"

# None = parse the entire selected dataset
RST_LIMIT = None
RST_REBUILD = False

# Whether to retry previously failed documents
RST_RETRY_ERRORS = False

# ------------------------------------------------------------
# XML forest configuration
# ------------------------------------------------------------

# Generate one forest per unique combination
FOREST_GROUP_BY = [
    "genre",
    "source",
    "n_sentences",
]

# False = reuse existing forests
# True  = regenerate all forests
FOREST_REBUILD = False

# Validate existing XML forests before reusing
FOREST_VALIDATE = True

# None = extract metadata automatically
FOREST_METADATA_FILE = None

# ------------------------------------------------------------
# TreeRePair compression configuration
# ------------------------------------------------------------

# None = use DOMAIN
COMPRESSION_GENRE = None

# None = use N_SENTENCES
COMPRESSION_N_SENTENCES = None

# None = compress all available sources
COMPRESSION_SOURCES = None

# False = reuse matching cached grammars
# True  = recompress all selected forests
COMPRESSION_REBUILD = False

# Generate the human-readable grammar (.reptxt)
COMPRESSION_TEXTUAL = True

# TreeRePair optimization mode
COMPRESSION_OPTIMIZE = "filesize"

# None = use TreeRePair default
COMPRESSION_MAX_RANK = None

# Maximum execution time per forest, in seconds
COMPRESSION_TIMEOUT = 600

# Continue processing other forests after an error
COMPRESSION_CONTINUE_ON_ERROR = True

# ------------------------------------------------------------
# TreeDePair validation configuration
# ------------------------------------------------------------

# False = reuse cached validation results
# True  = validate all selected grammars again
VALIDATION_REBUILD = False

# Maximum execution time per grammar, in seconds
VALIDATION_TIMEOUT = 600

# Keep decompressed XML files after validation
VALIDATION_KEEP_RECONSTRUCTED = True

# ------------------------------------------------------------
# RST motif extraction configuration
# ------------------------------------------------------------

# Root directory for motif results
MOTIF_RESULTS_DIR = None  # None = RESULTS_DIR / "motifs"

# False = reuse cached motifs if inputs and options match
# True  = regenerate all selected motifs
MOTIF_REBUILD = False

# Minimum number of rhetorical relations
MOTIF_MIN_RELATIONS = 1

# Minimum root-to-leaf depth measured in rhetorical edges
MOTIF_MIN_DEPTH = 1

# Include both parameterized and fully instantiated motifs
MOTIF_INCLUDE_WILDCARDS = True

# Continue processing other grammars after an error
MOTIF_CONTINUE_ON_ERROR = True

# Show grammar symbols in compact motif tables
MOTIF_SHOW_GRAMMAR_SYMBOLS = True