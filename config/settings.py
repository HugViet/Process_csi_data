import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WEIGHTS_DIR = DATA_DIR / "weights"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

SERIAL_PORT = os.getenv("CSI_SERIAL_PORT", "COM3")
BAUD_RATE = int(os.getenv("CSI_BAUD_RATE", 921600))
NUM_SUBCARRIERS = 64
