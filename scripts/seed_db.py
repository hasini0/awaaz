"""Fill the demo database (the web app also does this automatically on first run).

    python scripts/seed_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from awaaz.demo import seed

if __name__ == "__main__":
    store = seed()
    print(f"seeded {store.count()} complaints into {store.path}")
