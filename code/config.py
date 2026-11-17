"""Central paths and constants for the RAID rhetorical-signatures pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "raid"
TRAIN_FILE = RAW_DIR / "train.csv"

# Shared sentence-analysis cache and selected datasets are separate.
RESULTS_DIR = PROJECT_ROOT / "results" / "sentences_report"
FILTERED_DIR = DATA_DIR / "filtered"
MATCHED_DIR = FILTERED_DIR / "matched"
UNMATCHED_DIR = FILTERED_DIR / "unmatched"

DATASET_ID = "liamdugan/raid"
CHUNK_SIZE = 100_000
BATCH_SIZE = 256
RANDOM_SEED = 42
LENGTH_BINS = 10
DEFAULT_N_SENTENCES = 6  # Compatibility for other existing modules.
SPACY_MODEL = "en_core_web_sm"

DOMAIN_NAMES = {
    "abstracts": "ArXiv Abstracts",
    "reddit": "Reddit Posts",
    "books": "Book Summaries",
    "news": "NYT News Articles",
    "reviews": "Movie Reviews",
    "poetry": "Poetry",
    "recipes": "Recipes",
}
MODEL_NAMES = {
    "human": "Human",
    "chatgpt": "ChatGPT",
    "gpt4": "GPT-4",
    "mistral": "Mistral 7B",
    "mistral-chat": "Mistral 7B (Chat)",
    "cohere": "Cohere",
    "cohere-chat": "Cohere (Chat)",
}
MODEL_ORDER = list(MODEL_NAMES.values())


def locate_training_file() -> Path:
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(
            f"RAID train.csv not found at {TRAIN_FILE}. "
            "Run 01_download_raid.py first."
        )
    return TRAIN_FILE
