"""Safety rules: a deterministic layer that can only raise urgency, never lower it."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from . import config
from .text import normalize


@dataclass(frozen=True)
class RuleHit:
    rule_id: str
    level: str
    reason: str
    keyword: str


@lru_cache
def _compiled():
    rules = []
    for r in config.safety_rules():
        pats = []
        for keyword in r["keywords"]:
            kw = normalize(str(keyword)).lower()
            if kw.isascii():
                # word boundaries for Latin text so "fire" doesn't match "fireworks"-style substrings
                pats.append((kw, re.compile(rf"\b{re.escape(kw)}\b")))
            else:
                pats.append((kw, re.compile(re.escape(kw))))
        rules.append((r, pats))
    return rules


def check(text: str) -> list[RuleHit]:
    t = normalize(text).lower()
    hits = []
    for r, pats in _compiled():
        for kw, pat in pats:
            if pat.search(t):
                hits.append(RuleHit(r["id"], r["level"], r["reason"], kw))
                break
    return hits


def apply(level: str, hits: list[RuleHit]) -> str:
    order = config.urgency_ids()
    best = order.index(level)
    for h in hits:
        best = max(best, order.index(h.level))
    return order[best]
