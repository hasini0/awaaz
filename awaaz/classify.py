"""Department + urgency classifiers. Three interchangeable backends share one interface:

* TfidfClassifier  — char/word n-gram logistic regression (baseline; also used for explanations)
* EmbedClassifier  — multilingual-e5 sentence embeddings + logistic regression heads (fast, deployable)
* EnsembleClassifier — average of TF-IDF and e5 probabilities
* MurilClassifier  — google/muril-base-cased fine-tuned with two heads (needs ~1GB)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from . import config
from .locate import default_locator

STOPWORDS = set("""the a an in on at of to and or is are was were be been has have had it this that our we my i
near for from with by sir madam please kindly loc since days day week weeks month there here very
में है हैं का की के से को और पर यह हमारी हमारे सर ने
లో మా ఉంది ఉన్నాయి సార్ ఈ ఆ కూడా
میں ہے ہیں کا کی کے سے کو اور پر یہ
lo maa undi mein hai hain ka ki ke se ko aur pe par ye""".split())


# \w misses Indic vowel signs (Unicode Mc/Mn), which splits "కరెంట్" into fragments — include the blocks.
WORD_PATTERN = r"(?u)[\w\u0900-\u097F\u0C00-\u0C7F\u0600-\u06FF\u0750-\u077F]+"


def prep(text: str) -> str:
    """Normalise + mask locality names. Every backend sees text through this function."""
    return default_locator().mask(text)


@dataclass
class Prediction:
    department: str
    dept_conf: float
    urgency: str
    urg_conf: float
    dept_probs: dict = field(default_factory=dict)
    urg_probs: dict = field(default_factory=dict)


def _preds(dept_classes, dp, urg_classes, up) -> list[Prediction]:
    out = []
    for d, u in zip(dp, up):
        di, ui = int(np.argmax(d)), int(np.argmax(u))
        out.append(Prediction(dept_classes[di], float(d[di]), urg_classes[ui], float(u[ui]),
                              dict(zip(dept_classes, map(float, d))), dict(zip(urg_classes, map(float, u)))))
    return out


# ------------------------------------------------------------------------------- TF-IDF
class TfidfClassifier:
    name = "tfidf"

    def __init__(self):
        self.vec = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     token_pattern=WORD_PATTERN)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                     max_features=200_000)),
        ])
        self.dept = LogisticRegression(max_iter=2000, C=8)
        self.urg = LogisticRegression(max_iter=2000, C=8, class_weight="balanced")

    def fit(self, texts, depts, urgs):
        X = self.vec.fit_transform([prep(t) for t in texts])
        self.dept.fit(X, depts)
        self.urg.fit(X, urgs)
        return self

    def predict(self, texts) -> list[Prediction]:
        X = self.vec.transform([prep(t) for t in texts])
        return _preds(list(self.dept.classes_), self.dept.predict_proba(X),
                      list(self.urg.classes_), self.urg.predict_proba(X))

    def explain(self, text: str, department: str, k: int = 5) -> list[str]:
        """Top words that pushed the model toward `department` (word features only, readable)."""
        word_vec: TfidfVectorizer = self.vec.transformer_list[0][1]
        x = word_vec.transform([prep(text)])
        n_word = len(word_vec.vocabulary_)
        ci = list(self.dept.classes_).index(department)
        contrib = x.multiply(self.dept.coef_[ci, :n_word]).tocsr()
        names = word_vec.get_feature_names_out()
        idx = contrib.indices[np.argsort(-contrib.data)]
        words = [names[i] for i in idx
                 if contrib[0, i] > 0 and not all(w in STOPWORDS or w.isdigit() for w in names[i].split())]
        out = []
        for w in words:
            if not any(w in o or o in w for o in out):
                out.append(w)
            if len(out) == k:
                break
        return out

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.joblib", compress=3)

    @staticmethod
    def load(path: Path) -> TfidfClassifier:
        return joblib.load(path / "model.joblib")


# -------------------------------------------------------------------------- Embeddings
class Embedder:
    """multilingual-e5 sentence embeddings (also used for duplicate detection)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.settings()["embedder"]["model"]
        self.model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        prefix = "query: " if "e5" in self.model_name else ""
        return self.model.encode([prefix + prep(t) for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


@lru_cache
def default_embedder() -> Embedder:
    return Embedder()


class EmbedClassifier:
    name = "embed"

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder
        self.dept = LogisticRegression(max_iter=3000, C=20)
        self.urg = LogisticRegression(max_iter=3000, C=20, class_weight="balanced")

    def _emb(self):
        if self.embedder is None:
            self.embedder = default_embedder()
        return self.embedder

    def fit(self, texts, depts, urgs, X: np.ndarray | None = None):
        X = self._emb().encode(texts) if X is None else X
        self.dept.fit(X, depts)
        self.urg.fit(X, urgs)
        return self

    def predict(self, texts, X: np.ndarray | None = None) -> list[Prediction]:
        X = self._emb().encode(texts) if X is None else X
        return _preds(list(self.dept.classes_), self.dept.predict_proba(X),
                      list(self.urg.classes_), self.urg.predict_proba(X))

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"dept": self.dept, "urg": self.urg, "embedder": self._emb().model_name},
                    path / "heads.joblib", compress=3)

    @staticmethod
    def load(path: Path) -> EmbedClassifier:
        d = joblib.load(path / "heads.joblib")
        c = EmbedClassifier(default_embedder())
        c.dept, c.urg = d["dept"], d["urg"]
        return c


# ------------------------------------------------------------------------------- MuRIL
class MurilClassifier:
    name = "muril"

    def __init__(self, path: Path):
        import torch
        from transformers import AutoTokenizer

        from .muril import MultiTaskModel

        meta = json.loads((path / "meta.json").read_text())
        self.depts, self.urgs, self.max_len = meta["departments"], meta["urgencies"], meta["max_len"]
        # temperature scaling (fitted on the validation split) turns raw logits into calibrated confidences
        self.t_dept, self.t_urg = meta.get("temperature_dept", 1.0), meta.get("temperature_urg", 1.0)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = MultiTaskModel(meta["base_model"], len(self.depts), len(self.urgs), from_pretrained=False,
                                    config_dir=path)
        self.model.load_state_dict(torch.load(path / "heads_and_encoder.pt", map_location="cpu"))
        self.model.eval()
        self.torch = torch

    def logits(self, texts):
        """Raw (uncalibrated) logits for both heads — used to fit the temperatures."""
        dls, uls = [], []
        with self.torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self.tok([prep(t) for t in texts[i:i + 32]], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                dl, ul = self.model(enc["input_ids"], enc["attention_mask"])
                dls.append(dl)
                uls.append(ul)
        return self.torch.cat(dls), self.torch.cat(uls)

    def predict(self, texts) -> list[Prediction]:
        dl, ul = self.logits(texts)
        return _preds(self.depts, (dl / self.t_dept).softmax(-1).numpy(),
                      self.urgs, (ul / self.t_urg).softmax(-1).numpy())


# ---------------------------------------------------------------------------- Ensemble
class EnsembleClassifier:
    """Equal-weight average of member probabilities. The lexical model (TF-IDF) is precise on familiar
    wording; the semantic model (e5) generalises to new phrasing — averaging gets both."""
    name = "ensemble"

    def __init__(self, members: list):
        self.members = members

    def predict(self, texts) -> list[Prediction]:
        per_member = [m.predict(texts) for m in self.members]
        out = []
        for preds in zip(*per_member):
            dept = {k: float(np.mean([p.dept_probs[k] for p in preds])) for k in preds[0].dept_probs}
            urg = {k: float(np.mean([p.urg_probs[k] for p in preds])) for k in preds[0].urg_probs}
            d, u = max(dept, key=dept.get), max(urg, key=urg.get)
            out.append(Prediction(d, dept[d], u, urg[u], dept, urg))
        return out


# ------------------------------------------------------------------------------ loader
def available_backends() -> list[str]:
    found = []
    if (config.MODELS_DIR / "tfidf" / "model.joblib").exists():
        found.append("tfidf")
    if (config.MODELS_DIR / "embed" / "heads.joblib").exists():
        found.append("embed")
    if {"tfidf", "embed"} <= set(found):
        found.append("ensemble")
    if (config.MODELS_DIR / "muril" / "heads_and_encoder.pt").exists():
        found.append("muril")
    return found


def best_backend() -> str:
    """Pick the backend with the best macro-F1 on the held-out *unseen-phrasing* split. The hand-written
    set is never used for selection, so it stays an untouched final test."""
    found = available_backends()
    if not found:
        raise FileNotFoundError("No trained models in models/. Run: python scripts/train.py")
    metrics_file = config.REPORTS_DIR / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())["models"]
        split = "synthetic_unseen_phrasing"
        scored = [(m[b][split]["dept_macro_f1"] + m[b][split]["urg_macro_f1"], b)
                  for b in found if b in m]
        if scored:
            return max(scored)[1]
    return found[-1]


@lru_cache
def load_classifier(backend: str = "auto"):
    if backend == "auto":
        backend = config.settings()["classifier"]["backend"]
    if backend == "auto":
        backend = best_backend()
    if backend == "ensemble":
        return EnsembleClassifier([load_classifier("tfidf"), load_classifier("embed")])
    path = config.MODELS_DIR / backend
    return {"tfidf": TfidfClassifier.load, "embed": EmbedClassifier.load, "muril": MurilClassifier}[backend](path)


@lru_cache
def load_explainer() -> TfidfClassifier | None:
    p = config.MODELS_DIR / "tfidf"
    return TfidfClassifier.load(p) if (p / "model.joblib").exists() else None
