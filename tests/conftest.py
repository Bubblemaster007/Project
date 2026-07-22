from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOPIC2_REMAINING = ROOT / "topic2_remaining_code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOPIC2_REMAINING) not in sys.path:
    sys.path.insert(0, str(TOPIC2_REMAINING))
