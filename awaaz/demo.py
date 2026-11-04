"""Demo data: ~3 weeks of realistic complaints, including planted hotspots where many
citizens report the same problem in different languages."""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from . import config, synth
from .pipeline import Awaaz
from .store import Store

# (department, issue-clause index, locality, reports, spread over N days)
HOTSPOTS = [
    ("sewerage", 0, "kukatpally", 18, 5),
    ("water", 0, "dilsukhnagar", 12, 4),
    ("garbage", 1, "charminar", 10, 6),
    ("stray_animals", 0, "gachibowli", 8, 7),
    ("mosquito", 1, "malakpet", 7, 5),
    ("electricity", 1, "ameerpet", 4, 1),
    ("roads", 1, "miyapur", 6, 8),
]
BACKGROUND = 260
CHANNELS = ["voice", "voice", "web", "whatsapp", "ivr"]


def _hotspot_reports(rng: random.Random, now: datetime) -> list[tuple[datetime, str, str | None]]:
    localities = {loc["id"]: loc for loc in config.city()["localities"]}
    items = []
    for dept, idx, loc_id, n, spread in HOTSPOTS:
        langs = [lang for lang in synth.LANGS if idx < len(synth.ISSUES[dept][lang])]
        for _ in range(n):
            r = synth.make_one(dept, rng.choice(langs), idx, rng, [localities[loc_id]], noise=0.1)
            items.append((now - timedelta(days=rng.uniform(0, spread), hours=rng.uniform(0, 3)), r["text"], loc_id))
    return items


def _background_reports(rng: random.Random, now: datetime, seed: int) -> list[tuple[datetime, str, str | None]]:
    rows = synth.generate(BACKGROUND, "train", seed=seed) + synth.generate(BACKGROUND // 3, "test", seed=seed + 1)
    # locality=None: the pipeline extracts the location from the text itself
    return [(now - timedelta(days=rng.uniform(0, 21), hours=rng.uniform(0, 12)), r["text"], None) for r in rows]


def _age_statuses(store: Store, rng: random.Random, now: datetime):
    """Incidents past their department's deadline were mostly resolved; the planted hotspots and anything
    still within its deadline are open or being worked on, so the dashboard has a story to tell."""
    df = store.df()
    planted = {(dept, loc) for dept, _, loc, _, _ in HOTSPOTS}
    df["planted"] = [(d, loc) in planted for d, loc in zip(df["department"], df["locality_id"])]
    # a planted report can merge into an earlier report from a neighbouring locality, so check every member
    clusters = df.groupby("cluster_id").agg(first=("created_at", "min"), department=("department", "first"),
                                            planted=("planted", "any"))
    for cid, c in clusters.iterrows():
        sla_ratio = (now - c["first"]).total_seconds() / 3600 / config.departments()[c["department"]].sla_hours
        if c["planted"] or sla_ratio < 1:
            if rng.random() < 0.25:
                store.set_cluster_status(cid, "in_progress")
        elif rng.random() < 0.93:
            store.set_cluster_status(cid, "resolved")


def seed(engine: Awaaz | None = None, seed: int = 42) -> Store:
    """Replace the engine's database with the demo data. Pass the running engine to reuse its loaded models."""
    rng = random.Random(seed)
    engine = engine or Awaaz()
    store = engine.store
    store.clear()
    now = datetime.now()

    items = sorted(_hotspot_reports(rng, now) + _background_reports(rng, now, seed))
    for when, text, loc_id in items:  # chronological, so duplicate detection sees history in order
        engine.submit(text, channel=rng.choice(CHANNELS), locality_id=loc_id, created_at=when)

    _age_statuses(store, rng, now)
    return store
