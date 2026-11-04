"""End-to-end: voice/text -> understood, located, de-duplicated, prioritised ticket."""
from __future__ import annotations

import math
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import asr, config, rules
from .classify import available_backends, default_embedder, load_classifier, load_explainer
from .cluster import find_duplicate
from .locate import Location, default_locator, jitter
from .store import Store
from .text import LANG_NAMES, detect_language, normalize


@dataclass
class Ticket:
    id: str
    created_at: str
    text: str
    lang: str
    lang_name: str
    department: str
    department_name: str
    agency: str
    sla_hours: int
    dept_conf: float
    urgency: str
    model_urgency: str
    urg_conf: float
    rule_hits: list = field(default_factory=list)
    keywords: list = field(default_factory=list)
    locality_id: str | None = None
    locality: str | None = None
    lat: float | None = None
    lon: float | None = None
    location_method: str | None = None
    cluster_id: str | None = None
    reports: int = 1
    duplicate_similarity: float | None = None
    priority: float = 0.0
    needs_review: bool = False
    routing_unsure: bool = False  # the second-opinion models disagree or are unsure about the department
    department_by_citizen: bool = False  # the citizen picked the department when the chatbot asked
    second_opinions: dict = field(default_factory=dict)  # backend -> department it predicted
    timings_ms: dict = field(default_factory=dict)
    dept_probs: dict = field(default_factory=dict)
    urg_probs: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def priority_score(urgency: str, reports: int) -> float:
    """0-100. Severity dominates; community corroboration (duplicate reports) adds up to 20."""
    w = config.urgency(urgency).weight
    return round(min(100.0, 20 * w + min(20.0, 6 * math.log2(max(reports, 1)))), 1)


class Awaaz:
    REVIEW_THRESHOLD = 0.55  # below this department confidence a human should check the routing
    # Fast models that give a second opinion on the department. A fine-tuned model is confidently wrong on
    # vague text ("big problem near my house"); when the committee disagrees, we ask the citizen instead.
    # On the hand-written set the main model was right on 100% of complaints where all models agreed.
    SECOND_OPINIONS = ("tfidf", "embed")

    def __init__(self, store: Store | None = None, backend: str = "auto"):
        self.store = store or Store()
        self.clf = load_classifier(backend)
        self.checkers = [load_classifier(b) for b in self.SECOND_OPINIONS
                         if b in available_backends() and b != self.clf.name]
        self.explainer = load_explainer()
        self.embedder = default_embedder()
        self.locator = default_locator()

    @property
    def backend(self) -> str:
        return self.clf.name

    def transcribe(self, audio_path: str, language: str | None = None) -> dict:
        return asr.transcribe(audio_path, language)

    def analyze(self, text: str, locality_id: str | None = None, department: str | None = None) -> Ticket:
        """Understand a complaint without saving it. `locality_id` / `department` override the AI when the
        citizen has told us (e.g. picked the department when the chatbot asked)."""
        text = normalize(text)
        if not text:
            raise ValueError("Complaint text is empty.")
        if locality_id and locality_id not in self.locator.by_id:
            raise ValueError(f"Unknown locality_id {locality_id!r}.")
        if department and department not in config.departments():
            raise ValueError(f"Unknown department {department!r}.")
        t = {}
        s = time.perf_counter()
        lang = detect_language(text)
        pred = self.clf.predict([text])[0]
        opinions = {} if department else {c.name: c.predict([text])[0] for c in self.checkers}
        if department:
            pred.department, pred.dept_conf = department, 1.0
        support = [pred.dept_conf] + [o.dept_probs.get(pred.department, 0.0) for o in opinions.values()]
        unsure = (any(o.department != pred.department for o in opinions.values())
                  or sum(support) / len(support) < self.REVIEW_THRESHOLD)
        t["classify"] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        hits = rules.check(text)
        urgency = rules.apply(pred.urgency, hits)
        t["rules"] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        loc: Location | None = self.locator.get(locality_id) if locality_id else self.locator.find(text)
        t["locate"] = (time.perf_counter() - s) * 1000

        dept = config.departments()[pred.department]
        keywords = self.explainer.explain(text, pred.department) if self.explainer else []
        return Ticket(
            id="", created_at="", text=text, lang=lang, lang_name=LANG_NAMES.get(lang, lang),
            department=dept.id, department_name=dept.name, agency=dept.agency, sla_hours=dept.sla_hours,
            dept_conf=round(pred.dept_conf, 3), urgency=urgency, model_urgency=pred.urgency,
            urg_conf=round(pred.urg_conf, 3), rule_hits=[asdict(h) for h in hits], keywords=keywords,
            locality_id=loc.id if loc else None, locality=loc.name if loc else None,
            lat=loc.lat if loc else None, lon=loc.lon if loc else None,
            location_method=loc.method if loc else None,
            needs_review=unsure or loc is None, routing_unsure=unsure, department_by_citizen=bool(department),
            second_opinions={k: o.department for k, o in opinions.items()},
            timings_ms={k: round(v, 1) for k, v in t.items()},
            dept_probs=pred.dept_probs, urg_probs=pred.urg_probs,
        )

    def submit(self, text: str, channel: str = "web", locality_id: str | None = None,
               created_at: datetime | None = None, department: str | None = None) -> Ticket:
        """Understand, de-duplicate, prioritise and store a complaint."""
        tk = self.analyze(text, locality_id, department)
        now = created_at or datetime.now()
        tk.id = f"AWZ-{now:%y%m}-{secrets.token_hex(3).upper()}"
        tk.created_at = now.isoformat(timespec="seconds")

        s = time.perf_counter()
        emb = self.embedder.encode([tk.text])[0]
        if tk.lat is not None:
            tk.lat, tk.lon = jitter(tk.lat, tk.lon, tk.id)
        dup = find_duplicate(self.store, tk.department, tk.lat, tk.lon, emb, now)
        tk.timings_ms["dedup"] = round((time.perf_counter() - s) * 1000, 1)

        if dup:
            tk.cluster_id, tk.reports, tk.duplicate_similarity = dup.cluster_id, dup.reports, round(dup.similarity, 3)
            # an incident is as severe as its most severe report, plus corroboration from every report
            levels = config.urgency_ids()
            worst = max([tk.urgency, *self.store.cluster_urgencies(dup.cluster_id)], key=levels.index)
            tk.priority = priority_score(worst, tk.reports)
        else:
            tk.cluster_id, tk.reports = tk.id, 1
            tk.priority = priority_score(tk.urgency, 1)

        self.store.insert({
            "id": tk.id, "created_at": tk.created_at, "channel": channel, "text": tk.text, "lang": tk.lang,
            "department": tk.department, "dept_conf": tk.dept_conf, "urgency": tk.urgency,
            "urg_conf": tk.urg_conf, "model_urgency": tk.model_urgency, "rule_hits": tk.rule_hits,
            "locality_id": tk.locality_id, "locality": tk.locality, "lat": tk.lat, "lon": tk.lon,
            "cluster_id": tk.cluster_id, "priority": tk.priority, "embedding": emb,
            # a report joining an incident officers already started on is "in progress" too
            "status": self.store.cluster_status(tk.cluster_id) if dup else "open",
        })
        if dup:
            self.store.set_cluster_priority(tk.cluster_id, tk.priority)
        return tk
