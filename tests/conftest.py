import sys
from pathlib import Path

# Ensure src on path for tests/CLIs when not installed
ROOT = Path(__file__).resolve().parents[1]  # liq-examples
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Add sibling library paths needed for imports when running from monorepo
quant_root = ROOT.parent
SIBLINGS = [
    quant_root / "liq-metrics" / "src",
    quant_root / "liq-features" / "src",
    quant_root / "liq-data" / "src",
    quant_root / "liq-sim" / "src",
    quant_root / "liq-signals" / "src",
    quant_root / "liq-store" / "src",
    quant_root / "liq-risk" / "src",
]
for path in SIBLINGS:
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
