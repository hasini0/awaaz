"""Duplicate detection (same issue, same place) and geographic hotspot discovery."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from . import config
from .locate import haversine_km
from .store import ACTIVE, Store

EARTH_KM = 6371.0


@dataclass
class DuplicateMatch:
    cluster_id: str
    similarity: float
    distance_km: float
    reports: int  # including the new one


def find_duplicate(store: Store, department: str, lat: float | None, lon: float | None, emb: np.ndarray,
                   now: datetime | None = None) -> DuplicateMatch | None:
    """A new complaint joins an existing cluster when it is the same department, nearby, and
    semantically similar (multilingual embeddings, so a Telugu and a Hindi report of the same
    overflowing drain are matched)."""
    if lat is None:
        return None
    cfg = config.settings()["dedup"]
    now = now or datetime.now()
    best = None
    for row in store.candidates(department, now - timedelta(days=cfg["window_days"])):
        d = haversine_km(lat, lon, row["lat"], row["lon"])
        if d > cfg["radius_km"] or row["embedding"] is None:
            continue
        sim = float(np.frombuffer(row["embedding"], dtype=np.float32) @ emb)
        if sim >= cfg["similarity"] and (best is None or sim > best[0]):
            best = (sim, d, row["cluster_id"])
    if not best:
        return None
    sim, d, cid = best
    return DuplicateMatch(cid, sim, d, store.cluster_size(cid) + 1)


def hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """DBSCAN (haversine) over unresolved complaints, per department. Returns one row per hotspot."""
    cfg = config.settings()["hotspots"]
    rows = []
    open_df = df[df["status"].isin(ACTIVE) & df["lat"].notna()]
    for dept, g in open_df.groupby("department"):
        if len(g) < cfg["min_reports"]:
            continue
        coords = np.radians(g[["lat", "lon"]].to_numpy())
        labels = DBSCAN(eps=cfg["eps_km"] / EARTH_KM, min_samples=cfg["min_reports"], metric="haversine",
                        algorithm="ball_tree").fit_predict(coords)
        for lab in set(labels) - {-1}:
            m = g[labels == lab]
            levels = config.urgency_ids()
            top = max(m["urgency"], key=levels.index)
            rows.append({
                "department": dept, "reports": len(m), "lat": m["lat"].mean(), "lon": m["lon"].mean(),
                "localities": ", ".join(m["locality"].dropna().value_counts().index[:3]),
                "max_urgency": top, "first_seen": m["created_at"].min(), "last_seen": m["created_at"].max(),
                "priority": m["priority"].max(),
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["priority", "reports"], ascending=False) if len(out) else out


def spikes(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Early warning: (department, locality) pairs reporting far more than their own usual rate.
    Compares the last `window_hours` with the daily average of the `baseline_days` before it."""
    cfg = config.settings()["spikes"]
    now = now or datetime.now()
    window_start = now - timedelta(hours=cfg["window_hours"])
    base_start = window_start - timedelta(days=cfg["baseline_days"])
    d = df[df["locality_id"].notna() & (df["created_at"] >= base_start)]
    rows = []
    for (dept, _), g in d.groupby(["department", "locality_id"]):
        recent = int((g["created_at"] >= window_start).sum())
        if recent < cfg["min_reports"]:
            continue
        per_day_recent = recent / (cfg["window_hours"] / 24)
        per_day_usual = (len(g) - recent) / cfg["baseline_days"]
        if per_day_recent >= cfg["ratio"] * max(per_day_usual, cfg["floor_per_day"]):
            rows.append({"department": dept, "locality": g["locality"].iloc[0], "recent": recent,
                         "per_day_recent": round(per_day_recent, 1), "per_day_usual": round(per_day_usual, 2),
                         "critical": int((g["created_at"] >= window_start).mul(g["urgency"] == "critical").sum())})
    out = pd.DataFrame(rows)
    return out.sort_values(["critical", "recent"], ascending=False) if len(out) else out
