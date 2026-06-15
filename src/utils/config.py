import logging
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TIMESTAMP_YMD = datetime.now().strftime("%Y-%m-%d")
RUN_LOG_DIR = LOGS_DIR / TIMESTAMP_YMD
RUN_LOG_FILE = RUN_LOG_DIR / f"{TIMESTAMP}.log"

RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("medical_charges_prediction")

if not logger.handlers:
	logger.setLevel(logging.INFO)
	logger.addHandler(logging.StreamHandler())
	logger.addHandler(logging.FileHandler(RUN_LOG_FILE))


def log(*message_parts):
	message = " ".join(str(part) for part in message_parts)
	logger.info(message)
