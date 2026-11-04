"""Generate data, train the TF-IDF baseline and the multilingual-embedding model, evaluate both,
calibrate the duplicate-detection threshold, and write reports/metrics.json + figures.

    python scripts/train.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from awaaz import config, rules, synth
from awaaz.locate import Locator, haversine_km
from awaaz.classify import EmbedClassifier, EnsembleClassifier, TfidfClassifier, default_embedder

DATA = config.DATA_DIR
REPORTS = config.REPORTS_DIR
FIG = REPORTS / "figures"


def load_sets():
    DATA.mkdir(exist_ok=True)
    train = pd.DataFrame(synth.generate(8000, "train", seed=1))
    val = pd.DataFrame(synth.generate(1000, "train", seed=2))
    test = pd.DataFrame(synth.generate(2000, "test", seed=3))
    hand = pd.read_json(DATA / "test_handwritten.jsonl", lines=True)
    for name, d in [("train", train), ("val", val), ("test_synthetic", test)]:
        d.to_csv(DATA / f"{name}.csv", index=False)
    return train, val, test, hand


def with_rules(texts, urg_preds):
    return [rules.apply(u, rules.check(t)) for t, u in zip(texts, urg_preds)]


def evaluate(name, preds, df) -> dict:
    dp = [p.department for p in preds]
    up_model = [p.urgency for p in preds]
    up = with_rules(df["text"], up_model)
    order = config.urgency_ids()
    crit_p, crit_r, crit_f, _ = precision_recall_fscore_support(df["urgency"], up, labels=["critical"],
                                                                zero_division=0)
    # "under-triage": predicted urgency lower than truth — the dangerous error
    under = np.mean([order.index(p) < order.index(t) for p, t in zip(up, df["urgency"])])
    within1 = np.mean([abs(order.index(p) - order.index(t)) <= 1 for p, t in zip(up, df["urgency"])])
    res = {
        "n": len(df),
        "dept_accuracy": accuracy_score(df["department"], dp),
        "dept_macro_f1": f1_score(df["department"], dp, average="macro"),
        "urg_macro_f1_model_only": f1_score(df["urgency"], up_model, average="macro"),
        "urg_macro_f1": f1_score(df["urgency"], up, average="macro"),
        "urg_accuracy": accuracy_score(df["urgency"], up),
        "urg_within_one_level": float(within1),
        "critical_recall": float(crit_r[0]),
        "critical_precision": float(crit_p[0]),
        "under_triage_rate": float(under),
        "per_language": {},
    }
    for lang, g in df.groupby("lang"):
        idx = g.index
        res["per_language"][lang] = {
            "n": len(g),
            "dept_macro_f1": f1_score(g["department"], [dp[i] for i in idx], average="macro"),
            "urg_macro_f1": f1_score(g["urgency"], [up[i] for i in idx], average="macro"),
        }
    res["_dp"], res["_up"] = dp, up
    return res


def plot_cm(y, p, labels, title, path):
    cm = confusion_matrix(y, p, labels=labels, normalize="true")
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title)
    plt.ylabel("true")
    plt.xlabel("predicted")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def _cross_lingual_pairs(rng_seed: int = 0, per_cell: int = 6) -> pd.DataFrame:
    """Issue clauses 0-2 are written as translations of each other in every language, so two complaints
    built from the same (department, clause) are the same incident — even across languages."""
    import random
    rng = random.Random(rng_seed)
    locs = config.city()["localities"]
    rows = []
    for dept in synth.ISSUES:
        for idx in range(3):
            for lang in synth.LANGS:
                if (dept, lang, idx) == ("garbage", "ur", 2):  # the one clause that isn't a translation
                    continue
                for _ in range(per_cell):
                    rows.append({**synth.make_one(dept, lang, idx, rng, locs), "incident": f"{dept}/{idx}"})
    return pd.DataFrame(rows)


def calibrate_dedup(hand: pd.DataFrame, emb_fn) -> dict:
    """Pick the cosine threshold on synthetic cross-lingual pairs, then evaluate on the hand-written
    *parallel* set (the same incident described in different languages = duplicate)."""
    s = _cross_lingual_pairs()
    E = emb_fn(s["text"].tolist())
    sims, ys = [], []
    for i, j in itertools.combinations(range(len(s)), 2):
        if s.department[i] != s.department[j]:
            continue  # production only compares complaints routed to the same department
        sims.append(float(E[i] @ E[j]))
        ys.append(int(s.incident[i] == s.incident[j]))
    sims, ys = np.array(sims), np.array(ys)
    best_t, best_f = 0.85, -1
    for t in np.arange(0.70, 0.97, 0.005):
        f = f1_score(ys, sims >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f

    h = hand[hand["locality"] != ""].reset_index(drop=True)
    E = emb_fn(h["text"].tolist())
    where = {loc["id"]: (loc["lat"], loc["lon"]) for loc in config.city()["localities"]}
    radius = config.settings()["dedup"]["radius_km"]
    sim, dist, truth, cross = [], [], [], []
    for i, j in itertools.combinations(range(len(h)), 2):
        if h.department[i] != h.department[j]:
            continue
        sim.append(float(E[i] @ E[j]))
        dist.append(haversine_km(*where[h.locality[i]], *where[h.locality[j]]))
        truth.append(int(h.locality[i] == h.locality[j]))
        cross.append(h.lang[i] != h.lang[j])
    sim, dist, truth, cross = map(np.array, (sim, dist, truth, cross))

    def score(pred):
        p, r, f, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
        return {"precision": round(float(p), 3), "recall": round(float(r), 3), "f1": round(float(f), 3)}

    text_only = sim >= best_t
    full = text_only & (dist <= radius)  # what production does: same meaning AND nearby
    cross_pos = cross & (truth == 1)
    return {"threshold": round(best_t, 3), "synthetic_f1": round(best_f, 3), "pairs": int(len(truth)),
            "text_only": score(text_only), "text_plus_geo": score(full),
            "cross_language_recall": round(float(full[cross_pos].mean()), 3)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    train, val, test, hand = load_sets()
    print(f"train={len(train)} val={len(val)} test_synthetic={len(test)} handwritten={len(hand)}")
    print("urgency distribution (train):", train["urgency"].value_counts().to_dict())

    metrics = {"data": {"train": len(train), "val": len(val), "test_synthetic": len(test),
                        "test_handwritten": len(hand),
                        "languages": sorted(train["lang"].unique().tolist()),
                        "departments": sorted(train["department"].unique().tolist())},
               "models": {}}
    if (REPORTS / "metrics.json").exists():  # keep results written by train_muril.py / eval_voice.py
        old = json.loads((REPORTS / "metrics.json").read_text())
        if "muril" in old.get("models", {}):
            metrics["models"]["muril"] = old["models"]["muril"]
        if "voice" in old:
            metrics["voice"] = old["voice"]

    # ---------------------------------------------------------------- baseline
    t0 = time.time()
    tf = TfidfClassifier().fit(train["text"], train["department"], train["urgency"])
    tf.save(config.MODELS_DIR / "tfidf")
    print(f"tfidf trained in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- embeddings
    emb = default_embedder()
    t0 = time.time()
    Xtr = emb.encode(train["text"].tolist())
    ec = EmbedClassifier(emb).fit(train["text"], train["department"], train["urgency"], X=Xtr)
    ec.save(config.MODELS_DIR / "embed")
    print(f"embed trained in {time.time() - t0:.1f}s")

    displays = {"tfidf": "TF-IDF + LogReg (baseline)", "embed": "multilingual-e5 + LogReg",
                "ensemble": "Ensemble: TF-IDF + e5"}
    for name, model in [("tfidf", tf), ("embed", ec), ("ensemble", EnsembleClassifier([tf, ec]))]:
        m = {"display": displays[name]}
        for split, df in [("val", val), ("synthetic_unseen_phrasing", test), ("handwritten", hand)]:
            r = evaluate(name, model.predict(df["text"].tolist()), df.reset_index(drop=True))
            dp, up = r.pop("_dp"), r.pop("_up")
            m[split] = r
            if split == "handwritten":
                plot_cm(df["department"], dp, sorted(config.departments()), f"{name}: department (handwritten)",
                        FIG / f"cm_dept_{name}.png")
                plot_cm(df["urgency"], up, config.urgency_ids(), f"{name}: urgency (handwritten)",
                        FIG / f"cm_urg_{name}.png")
        # latency per complaint (single item, as in production)
        sample = hand["text"].tolist()[:40]
        t0 = time.perf_counter()
        for s in sample:
            model.predict([s])
        m["latency_ms_per_complaint"] = round((time.perf_counter() - t0) / len(sample) * 1000, 1)
        metrics["models"][name] = m
        h = m["handwritten"]
        print(f"[{name}] handwritten dept-F1={h['dept_macro_f1']:.3f} urg-F1={h['urg_macro_f1']:.3f} "
              f"critical-recall={h['critical_recall']:.3f} | unseen-phrasing dept-F1="
              f"{m['synthetic_unseen_phrasing']['dept_macro_f1']:.3f}")

    # ---------------------------------------------------------------- locality extraction
    locator = Locator()
    found = [(locator.find(t) or None) for t in hand["text"]]
    metrics["locality_accuracy"] = round(float(np.mean([(f.id if f else "") == want
                                                        for f, want in zip(found, hand["locality"])])), 3)
    print("locality accuracy:", metrics["locality_accuracy"])

    # ---------------------------------------------------------------- dedup
    metrics["dedup"] = calibrate_dedup(hand, emb.encode)
    print("dedup:", metrics["dedup"])
    s = (config.CONFIG_DIR / "settings.yaml").read_text()
    import re
    s = re.sub(r"similarity: [0-9.]+", f"similarity: {metrics['dedup']['threshold']}", s)
    (config.CONFIG_DIR / "settings.yaml").write_text(s)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("wrote", REPORTS / "metrics.json")


if __name__ == "__main__":
    main()
