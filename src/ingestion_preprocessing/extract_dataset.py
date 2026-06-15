from pathlib import Path
import shutil
import sys

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import log


RAW_DATA_DIR = PROJECT_ROOT / "database" / "raw"


def extract_dataset():
    path = Path(kagglehub.dataset_download("mirichoi0218/insurance"))
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in path.iterdir():
        destination = RAW_DATA_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    log("Path to dataset files:", path)
    log("Dataset copied to:", RAW_DATA_DIR)
    return RAW_DATA_DIR


if __name__ == "__main__":
    extract_dataset()