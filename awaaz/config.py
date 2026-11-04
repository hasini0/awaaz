"""Loads the YAML config. Everything city- or department-specific lives in config/, not in code."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class Department:
    id: str
    name: str
    agency: str
    sla_hours: int
    icon: str
    names: dict = field(default_factory=dict, hash=False)  # language code -> local name

    def name_in(self, lang: str) -> str:
        return self.names.get(lang, self.name)


@dataclass(frozen=True)
class UrgencyLevel:
    id: str
    label: str
    color: str
    weight: int


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def settings() -> dict:
    return _load(CONFIG_DIR / "settings.yaml")


@lru_cache
def _departments_file() -> dict:
    return _load(CONFIG_DIR / "departments.yaml")


@lru_cache
def departments() -> dict[str, Department]:
    raw = _departments_file()["departments"]
    return {k: Department(id=k, **v) for k, v in raw.items()}


@lru_cache
def urgency_levels() -> list[UrgencyLevel]:
    return [UrgencyLevel(**u) for u in _departments_file()["urgency_levels"]]


def urgency_ids() -> list[str]:
    return [u.id for u in urgency_levels()]


def urgency(level_id: str) -> UrgencyLevel:
    return next(u for u in urgency_levels() if u.id == level_id)


@lru_cache
def safety_rules() -> list[dict]:
    return _departments_file()["safety_rules"]


@lru_cache
def messages() -> dict:
    return _load(CONFIG_DIR / "messages.yaml")


@lru_cache
def city() -> dict:
    return _load(CONFIG_DIR / "cities" / f"{settings()['city']}.yaml")
