"""Put ``src`` on the import path so the scripts run from a clone with no install."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
CHECKPOINTS = REPO_ROOT / "checkpoints"
for _d in (RESULTS, FIGURES, CHECKPOINTS):
    _d.mkdir(parents=True, exist_ok=True)
