from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "raid"
DATASET_ID = "liamdugan/raid"


def download_raid():
    """Download RAID and record the exact dataset revision."""

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

    (DATASET_DIR / "download_revision.txt").write_text(
        f"dataset={DATASET_ID}\nrevision={revision}\n",
        encoding="utf-8",
    )

    print(f"Download completed: {DATASET_DIR}")


if __name__ == "__main__":
    download_raid()
