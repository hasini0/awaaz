"""End-to-end voice check: audio file -> language ID -> Whisper -> full pipeline, against known labels.
Clips are run *without* a language hint, exactly like "Auto-detect" in the app.

    python scripts/eval_voice.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from awaaz import asr, config
from awaaz.pipeline import Awaaz
from awaaz.store import Store

# clip -> (spoken language, department, locality)
LABELS = {
    "hi_sewage_kukatpally.wav": ("hi", "sewerage", "kukatpally"),
    "hi_live_wire_ameerpet.wav": ("hi", "electricity", "ameerpet"),
    "hi_no_water_dilsukhnagar.wav": ("hi", "water", "dilsukhnagar"),
    "en_dog_bite_gachibowli.wav": ("en", "stray_animals", "gachibowli"),
    "en_garbage_charminar.wav": ("en", "garbage", "charminar"),
    "te_sewage_kukatpally.wav": ("te", "sewerage", "kukatpally"),
    "te_live_wire_ameerpet.wav": ("te", "electricity", "ameerpet"),
    "te_no_water_dilsukhnagar.wav": ("te", "water", "dilsukhnagar"),
    "te_stray_dogs_gachibowli.wav": ("te", "stray_animals", "gachibowli"),
    "te_garbage_charminar.wav": ("te", "garbage", "charminar"),
}


def main():
    engine = Awaaz(Store(":memory:"))
    rows = []
    for clip, (lang, dept, loc) in LABELS.items():
        heard = asr.transcribe(str(config.DATA_DIR / "audio_samples" / clip))
        tk = engine.analyze(heard["text"])
        rows.append({"clip": clip, "heard": heard["text"], "asr_seconds": round(heard["latency_s"], 2),
                     "language_ok": heard["language"] == lang,
                     "department_ok": tk.department == dept, "locality_ok": tk.locality_id == loc,
                     "urgency": tk.urgency})
        marks = {k: "✓" if ok else "✗" for k, ok in (("lang", heard["language"] == lang),
                                                       ("dept", tk.department == dept), ("loc", tk.locality_id == loc))}
        print(f"{clip:32} " + " ".join(f"{k}={v}" for k, v in marks.items()) + f" | {heard['text']}")
    summary = {"clips": len(rows), "language_id_accuracy": sum(r["language_ok"] for r in rows) / len(rows),
               "department_accuracy": sum(r["department_ok"] for r in rows) / len(rows),
               "locality_accuracy": sum(r["locality_ok"] for r in rows) / len(rows),
               "mean_asr_seconds": round(sum(r["asr_seconds"] for r in rows) / len(rows), 2),
               "asr_model": config.settings()["asr"]["model"],
               "asr_overrides": config.settings()["asr"]["overrides"], "details": rows}
    f = config.REPORTS_DIR / "metrics.json"
    metrics = json.loads(f.read_text())
    metrics["voice"] = summary
    f.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print({k: v for k, v in summary.items() if k != "details"})


if __name__ == "__main__":
    main()
