"""Gazetteer-based locality extraction that works across Telugu, Hindi, Urdu and Latin script."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz import fuzz

from . import config
from .text import normalize, phonetic_skeleton


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    lat: float
    lon: float
    matched: str
    method: str  # exact | fuzzy | manual


class Locator:
    """Finds the locality a complaint is about.

    1. exact alias match in any script (Telugu/Hindi/Urdu names may carry case suffixes like "-లో")
    2. fuzzy match, tolerant to typos *and speech-recognition errors* ("अमीर पिट" → Ameerpet)
    3. phonetic-skeleton match for Indic scripts ("कुकतपली" → Kukatpally)
    """
    FUZZY_MIN = {"latin": 86, "native": 80}
    _TOKEN = re.compile(r"[a-z]+|[\u0900-\u0963\u0966-\u097F\u0C00-\u0C7F\u0620-\u065F\u0670-\u06D3]+")

    def __init__(self, city_cfg: dict | None = None):
        self.city = city_cfg or config.city()
        self.by_id = {loc["id"]: loc for loc in self.city["localities"]}
        self._exact = []  # (alias, compiled pattern, locality), longest alias first
        self._fuzzy = []  # (alias without spaces, locality)
        self._phonetic = []  # (consonant skeleton, locality) for Indic-script aliases
        for loc in self.city["localities"]:
            for alias in {loc["name"], *loc["aliases"]}:
                a = normalize(str(alias)).lower()
                # Latin names need word boundaries; native-script names are matched as substrings
                pat = re.compile(rf"\b{re.escape(a)}\b" if a.isascii() else re.escape(a))
                self._exact.append((a, pat, loc))
                self._fuzzy.append((a.replace(" ", ""), loc))
                if not a.isascii():
                    skel = phonetic_skeleton(a.replace(" ", ""))
                    if len(skel) >= 4:  # short skeletons are too ambiguous
                        self._phonetic.append((skel, loc))
        self._exact.sort(key=lambda x: -len(x[0]))

    def find(self, text: str) -> Location | None:
        t = normalize(text).lower()
        return self._exact_match(t) or self._fuzzy_match(t) or self._phonetic_match(t)

    def _exact_match(self, t: str) -> Location | None:
        compact = t.replace(" ", "")
        for alias, pat, loc in self._exact:
            if pat.search(t) or (not alias.isascii() and alias.replace(" ", "") in compact):
                return self._mk(loc, alias, "exact")
        return None

    def _fuzzy_match(self, t: str) -> Location | None:
        tokens = self._TOKEN.findall(t)
        grams = tokens + [a + b for a, b in zip(tokens, tokens[1:])]  # "अमीर पिट" → "अमीरपिट"
        best, best_score = None, 0.0
        for g in grams:
            for alias, loc in self._fuzzy:
                if len(alias) < 5 or abs(len(g) - len(alias)) > 4:
                    continue
                latin = alias.isascii()
                if latin != g.isascii():
                    continue
                # native scripts glue suffixes onto names ("ameerpetlo"), so also compare the prefix
                score = fuzz.ratio(g, alias) if latin else max(fuzz.ratio(g, alias), fuzz.ratio(g[:len(alias)], alias))
                if score >= self.FUZZY_MIN["latin" if latin else "native"] and score > best_score:
                    best, best_score = (loc, g), score
        return self._mk(best[0], best[1], "fuzzy") if best else None

    def _phonetic_match(self, t: str) -> Location | None:
        tokens = [w for w in self._TOKEN.findall(t) if not w.isascii()]
        for g in tokens + [a + b for a, b in zip(tokens, tokens[1:])]:
            skel = phonetic_skeleton(g)
            for alias_skel, loc in self._phonetic:
                if skel == alias_skel or (skel.startswith(alias_skel) and len(skel) - len(alias_skel) <= 2):
                    return self._mk(loc, g, "phonetic")
        return None

    def mask(self, text: str, token: str = "loc") -> str:
        """Lower-case the text and replace place names with `token`, so classifiers can't learn spurious
        place→department correlations (e.g. 'Kukatpally' always meaning drainage)."""
        masked = normalize(text).lower()
        for _, pattern, _ in self._exact:
            masked = pattern.sub(token, masked)
        return masked

    def get(self, loc_id: str) -> Location:
        loc = self.by_id[loc_id]
        return self._mk(loc, loc["name"], "manual")

    @staticmethod
    def _mk(loc: dict, matched: str, method: str) -> Location:
        return Location(loc["id"], loc["name"], loc["lat"], loc["lon"], matched, method)


def jitter(lat: float, lon: float, key: str, radius_m: float = 450) -> tuple[float, float]:
    """Deterministic offset so many complaints in one locality don't stack on a single pin."""
    h = hashlib.sha1(key.encode()).digest()
    r = radius_m * math.sqrt(h[0] / 255)
    theta = 2 * math.pi * h[1] / 255
    dlat = (r * math.cos(theta)) / 111_320
    dlon = (r * math.sin(theta)) / (111_320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


@lru_cache
def default_locator() -> Locator:
    return Locator()
