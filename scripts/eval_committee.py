"""How well "ask the citizen when the models disagree" works, on the hand-written test set.

    python scripts/eval_committee.py

Confident = the main model and the second-opinion models agree, so the chatbot files straight away.
Otherwise the chatbot offers two departments and the citizen picks one.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from awaaz import config
from awaaz.chat import ChatBot
from awaaz.pipeline import Awaaz
from awaaz.store import Store


def main():
    engine = Awaaz(Store(":memory:"))
    hw = pd.read_json(config.DATA_DIR / "test_handwritten.jsonl", lines=True)
    tickets = [engine.analyze(t) for t in hw["text"]]
    asked = [tk.routing_unsure for tk in tickets]
    n_asked = sum(asked)
    confident_ok = sum(tk.department == d for tk, d, a in zip(tickets, hw["department"], asked) if not a)
    offered_ok = sum(d in ChatBot._top_two(tk) for tk, d, a in zip(tickets, hw["department"], asked) if a)
    summary = {
        "main_model": engine.backend, "second_opinions": [c.name for c in engine.checkers], "complaints": len(hw),
        "asked_citizen": n_asked, "asked_rate": round(n_asked / len(hw), 3),
        "accuracy_when_confident": round(confident_ok / (len(hw) - n_asked), 3),
        "correct_option_offered_when_asked": round(offered_ok / max(n_asked, 1), 3),
        "main_model_accuracy": round(sum(tk.department == d for tk, d in zip(tickets, hw["department"])) / len(hw), 3),
        "routing_accuracy_with_one_tap": round((confident_ok + offered_ok) / len(hw), 3),
    }
    f = config.REPORTS_DIR / "metrics.json"
    metrics = json.loads(f.read_text())
    metrics["committee"] = summary
    f.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(summary)


if __name__ == "__main__":
    main()
