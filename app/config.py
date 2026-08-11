import os

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "model")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.keras")
LABELS_PATH = os.path.join(ARTIFACTS_DIR, "labels.json")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
