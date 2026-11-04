"""Fine-tune google/muril-base-cased with two heads (department + urgency).

    python scripts/train_muril.py               # train (~25 min on Apple M2, ~6 min on a T4) + evaluate
    python scripts/train_muril.py --eval-only   # re-calibrate + re-evaluate the saved model

Uses CUDA / Apple MPS when available. Writes models/muril/ and merges results into
reports/metrics.json under models.muril. Run scripts/train.py first (it writes the data splits).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from awaaz import config
from awaaz.classify import MurilClassifier, prep
from awaaz.muril import BASE_MODEL, MultiTaskModel
from scripts.train import FIG, REPORTS, evaluate, plot_cm

OUT = config.MODELS_DIR / "muril"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args) -> float:
    data = pd.read_csv(config.DATA_DIR / "train.csv")
    if args.limit:
        data = data.sample(args.limit, random_state=0)
    depts, urgs = sorted(config.departments()), config.urgency_ids()
    dev = device()
    print(f"device={dev} train={len(data)}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = MultiTaskModel(BASE_MODEL, len(depts), len(urgs)).to(dev)
    texts = [prep(t) for t in data["text"]]
    y_dept = torch.tensor([depts.index(d) for d in data["department"]])
    y_urg = torch.tensor([urgs.index(u) for u in data["urgency"]])
    loader = DataLoader(list(range(len(texts))), batch_size=args.batch, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for step, batch in enumerate(loader):
            enc = tok([texts[i] for i in batch], padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(dev)
            dept_logits, urg_logits = model(enc["input_ids"], enc["attention_mask"])
            loss = ce(dept_logits, y_dept[batch].to(dev)) + ce(urg_logits, y_urg[batch].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
        print(f"epoch {epoch} mean loss {total / len(loader):.4f}", flush=True)
    seconds = time.time() - t0

    OUT.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    torch.save(model.state_dict(), OUT / "heads_and_encoder.pt")
    tok.save_pretrained(OUT)
    model.encoder.config.save_pretrained(OUT)
    (OUT / "meta.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "departments": depts, "urgencies": urgs, "max_len": args.max_len,
        "epochs": args.epochs, "train_rows": len(data), "train_seconds": round(seconds)}))
    return seconds


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Single scalar T minimising validation NLL (Guo et al., 2017). Accuracy is unchanged; only confidence."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def _ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    """Expected calibration error: average gap between confidence and accuracy."""
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    return round(ece, 4)


def calibrate() -> dict:
    """Fit one temperature per head on the validation split and store it in meta.json."""
    clf = MurilClassifier(OUT)
    val = pd.read_csv(config.DATA_DIR / "val.csv")
    test = pd.read_csv(config.DATA_DIR / "test_synthetic.csv")
    report = {}
    temps = {}
    for split, df in [("val", val), ("test", test)]:
        dl, ul = clf.logits(df["text"].tolist())
        yd = torch.tensor([clf.depts.index(d) for d in df["department"]])
        yu = torch.tensor([clf.urgs.index(u) for u in df["urgency"]])
        if split == "val":
            temps = {"temperature_dept": _fit_temperature(dl, yd), "temperature_urg": _fit_temperature(ul, yu)}
        else:
            report = {
                "ece_dept_before": _ece(dl.softmax(-1), yd),
                "ece_dept_after": _ece((dl / temps["temperature_dept"]).softmax(-1), yd),
                "ece_urg_before": _ece(ul.softmax(-1), yu),
                "ece_urg_after": _ece((ul / temps["temperature_urg"]).softmax(-1), yu),
            }
    meta = json.loads((OUT / "meta.json").read_text())
    meta.update(temps)
    (OUT / "meta.json").write_text(json.dumps(meta))
    print("calibration:", {k: round(v, 3) for k, v in temps.items()}, report)
    return {**{k: round(v, 3) for k, v in temps.items()}, **report}


def evaluate_saved(calibration: dict):
    clf = MurilClassifier(OUT)
    meta = json.loads((OUT / "meta.json").read_text())
    m = {"display": "MuRIL fine-tuned (multi-task)", "train_seconds": meta.get("train_seconds"),
         "calibration": calibration}
    splits = [("val", pd.read_csv(config.DATA_DIR / "val.csv")),
              ("synthetic_unseen_phrasing", pd.read_csv(config.DATA_DIR / "test_synthetic.csv")),
              ("handwritten", pd.read_json(config.DATA_DIR / "test_handwritten.jsonl", lines=True))]
    for split, df in splits:
        r = evaluate("muril", clf.predict(df["text"].tolist()), df.reset_index(drop=True))
        dp, up = r.pop("_dp"), r.pop("_up")
        m[split] = r
        if split == "handwritten":
            plot_cm(df["department"], dp, sorted(config.departments()), "muril: department (handwritten)",
                    FIG / "cm_dept_muril.png")
            plot_cm(df["urgency"], up, config.urgency_ids(), "muril: urgency (handwritten)", FIG / "cm_urg_muril.png")

    sample = splits[2][1]["text"].tolist()[:40]
    t0 = time.perf_counter()
    for s in sample:
        clf.predict([s])
    m["latency_ms_per_complaint"] = round((time.perf_counter() - t0) / len(sample) * 1000, 1)

    metrics_file = REPORTS / "metrics.json"
    metrics = json.loads(metrics_file.read_text())
    metrics["models"]["muril"] = m
    metrics_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    h = m["handwritten"]
    print(f"[muril] handwritten dept-F1={h['dept_macro_f1']:.3f} urg-F1={h['urg_macro_f1']:.3f} "
          f"critical-recall={h['critical_recall']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="use only N training rows (quick runs)")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate models/muril")
    args = ap.parse_args()
    if not args.eval_only:
        train(args)
    evaluate_saved(calibrate())


if __name__ == "__main__":
    main()
