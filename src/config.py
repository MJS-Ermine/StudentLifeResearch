"""
Configuration settings for the StudentLife analysis project.
"""
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results"
FIGURES_ROOT = PROJECT_ROOT / "results" / "figures"
MODELS_ROOT = PROJECT_ROOT / "models"

# Create necessary directories
for directory in [DATA_ROOT, RESULTS_ROOT, FIGURES_ROOT, MODELS_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

# Study parameters
STUDY_WEEKS = 10
TARGET_VARIABLE = "gpa"
RANDOM_STATE = 42

# Data processing parameters
SLEEP_WINDOW = {
    "start": "22:00",
    "end": "06:00"
}

# Feature engineering parameters
FEATURE_PARAMS: Dict[str, Any] = {
    "sleep": {
        "min_duration": 4,  # hours
        "max_duration": 12,  # hours
    },
    "phone_usage": {
        "night_hours": ["22:00", "06:00"],
        "max_daily_usage": 12,  # hours
    },
    "activity": {
        "min_steps": 1000,
        "max_steps": 30000,
    }
}

# Model parameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": RANDOM_STATE
    },
    "ridge": {
        "alpha": 1.0,
        "random_state": RANDOM_STATE
    }
}