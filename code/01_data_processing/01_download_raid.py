
"""Download the RAID dataset if it is not already available."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "raid"

DATASET_ID = "liamdugan/raid"

TRAIN_FILE = DATASET_DIR / "train.csv"
REVISION_FILE = DATASET_DIR / "download_revision.txt"


def download_raid(force=False):
    """Ensure the RAID training dataset is available locally.

    If the training file already exists, reuse it unless force=True.
    Otherwise, download a pinned Hugging Face dataset revision.
    """

    if TRAIN_FILE.is_file() and not force:
        print(f"RAID dataset already available: {TRAIN_FILE}")
        return TRAIN_FILE

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    revision = api.repo_info(
        repo_id=DATASET_ID,
        repo_type="dataset",
    ).sha

    print(f"Dataset: {DATASET_ID}")
    print(f"Revision: {revision}")

    snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=DATASET_DIR,
        max_workers=4,
    )

    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(
            "RAID download completed, but the expected "
            f"training file was not found: {TRAIN_FILE}"
        )

    REVISION_FILE.write_text(
        f"dataset={DATASET_ID}\nrevision={revision}\n",
        encoding="utf-8",
    )

    print(f"RAID dataset ready: {TRAIN_FILE}")

    return TRAIN_FILE


if __name__ == "__main__":
    download_raid()