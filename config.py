from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).parent
DATA_ROOT: Path = PROJECT_ROOT / "data" / "dataset"
RESULTS_ROOT: Path = PROJECT_ROOT / "results"
FIGURES_ROOT: Path = PROJECT_ROOT / "results" / "figures"

RESULTS_ROOT.mkdir(exist_ok=True)
FIGURES_ROOT.mkdir(exist_ok=True)

STUDY_WEEKS: int = 10
TARGET_VARIABLE: str = "gpa"
RANDOM_STATE: int = 42 