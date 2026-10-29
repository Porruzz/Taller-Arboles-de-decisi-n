"""
Configuration module for the Decision Trees Workshop.
Contains project constants, random seeds, paths, and styling settings.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
FIGURES_DIR = BASE_DIR / "figures"
OUTPUT_DIR = BASE_DIR

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Random Seeds
RANDOM_STATE = 42

# Workshop Constants
POSITIVE_CLASS = "A"
SYNTHETIC_TARGET_COL = "Clase"
SYNTHETIC_FEATURE_COLS = ["Quilates", "Profundidad"]

# Iris Experiment Configuration
IRIS_CANDIDATE_DEPTHS = [1, 3, None]
IRIS_DEV_SIZE = 0.20
IRIS_VAL_SIZE_OF_DEV = 0.25  # 0.25 * 0.80 = 0.20 of total


