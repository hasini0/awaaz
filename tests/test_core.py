import json
from pathlib import Path

import pytest

from awaaz import rules, synth
from awaaz.locate import Locator, haversine_km, jitter
from awaaz.pipeline import priority_score
from awaaz.text import detect_language

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text,lang", [
    ("మా కాలనీలో నీళ్లు రావడం లేదు", "te"),
    ("पानी नहीं आ रहा है", "hi"),
    ("پانی نہیں آ رہا ہے", "ur"),
    ("There is no water supply in our street", "en"),
    ("maa colony lo neellu raavatledu", "tenglish"),
    ("hamare gali mein paani nahi aa raha hai", "hinglish"),
])
def test_language_detection(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text,loc", [
    ("కూకట్‌పల్లిలో డ్రైనేజీ పొంగుతోంది", "kukatpally"),
    ("दिलसुखनगर में पानी नहीं", "dilsukhnagar"),
    ("ٹولی چوکی میں کچرا", "tolichowki"),
    ("pothole in LB Nagar main road", "lb_nagar"),
    ("drain overflow in kukatpaly", "kukatpally"),  # misspelt -> fuzzy match
    ("Jubilee Hills road no 36", "jubilee_hills"),
    ("अमीर पिट में बिजली का तार गिरा", "ameerpet"),  # speech-recognition error -> fuzzy match
    ("कुकतपली बस्तोप के पास नाला", "kukatpally"),  # ASR spelling -> phonetic skeleton match
])
def test_locator(text, loc):
    found = Locator().find(text)
    assert found is not None and found.id == loc


def test_phonetic_skeleton():
    from awaaz.text import phonetic_skeleton
    assert phonetic_skeleton("कुकतपली") == phonetic_skeleton("कूकटपल्ली")


def test_locator_none():
    assert Locator().find("the road is broken") is None


def test_safety_rules_only_raise():
    hits = rules.check("A live wire has fallen on the road")
    assert any(h.rule_id == "electrocution" for h in hits)
    assert rules.apply("low", hits) == "critical"
    assert rules.apply("critical", []) == "critical"


def test_rules_no_false_positive_on_burning_lights():
    assert rules.check("Streetlights are burning during daytime") == []


def test_rules_hindi_aage_is_not_fire():
    assert rules.check("नाली जाम है, पानी आगे नहीं बह रहा") == []


def test_synthetic_holdout_is_disjoint():
    tr = {r["phrase"] for r in synth.generate(3000, "train", seed=1)}
    te = {r["phrase"] for r in synth.generate(1000, "test", seed=2)}
    assert tr.isdisjoint(te)


def test_handwritten_labels_valid():
    from awaaz import config
    lines = (ROOT / "data" / "test_handwritten.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) >= 100
    for r in rows:
        assert r["department"] in config.departments()
        assert r["urgency"] in config.urgency_ids()


def test_priority_monotonic():
    assert priority_score("critical", 1) > priority_score("high", 1) > priority_score("low", 1)
    assert priority_score("medium", 10) > priority_score("medium", 1)
    assert priority_score("critical", 1000) <= 100


def test_geo_helpers():
    lat, lon = jitter(17.4, 78.4, "x")
    assert haversine_km(17.4, 78.4, lat, lon) < 0.5


MODELS_READY = (ROOT / "models" / "tfidf" / "model.joblib").exists()


@pytest.mark.skipif(not MODELS_READY, reason="run scripts/train.py first")
@pytest.mark.parametrize("text,dept", [
    ("Uppal mein shaam se light nahi hai", "electricity"),        # Hinglish "light" = household power
    ("Kondapur lo street light veligatledu", "streetlights"),
    ("మా కాలనీలో నీళ్లు రావడం లేదు", "water"),
    ("कूकटपल्ली में नाला ओवरफ्लो हो रहा है", "sewerage"),
])
def test_baseline_routes_common_complaints(text, dept):
    from awaaz.classify import load_classifier
    assert load_classifier("tfidf").predict([text])[0].department == dept
