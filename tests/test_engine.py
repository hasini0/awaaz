"""Engine and API behaviour on a throwaway database (needs trained models: run scripts/train.py)."""
from pathlib import Path

import pytest

from awaaz.cluster import DuplicateMatch
from awaaz.text import detect_language

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "tfidf" / "model.joblib").exists(),
                                reason="run scripts/train.py first")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from awaaz.pipeline import Awaaz
    from awaaz.store import Store

    return Awaaz(Store(tmp_path_factory.mktemp("db") / "test.db"), backend="tfidf")


@pytest.fixture
def client(engine, monkeypatch):
    from fastapi.testclient import TestClient

    import api

    monkeypatch.setattr(api, "_engine", engine)
    return TestClient(api.app)


def test_english_with_sir_and_current_is_english():
    assert detect_language("Sir the current is not coming since morning") == "en"


def test_analyze_rejects_bad_input(engine):
    with pytest.raises(ValueError):
        engine.analyze("   ")
    with pytest.raises(ValueError):
        engine.analyze("no water", locality_id="atlantis")


def test_merged_incident_keeps_its_worst_urgency(engine, monkeypatch):
    """A mild follow-up report must not drop a critical incident's priority, and must add corroboration."""
    from awaaz import pipeline

    first = engine.submit("A live electric wire has fallen on the road in Ameerpet")
    assert first.urgency == "critical"
    monkeypatch.setattr(pipeline, "find_duplicate",
                        lambda *a, **k: DuplicateMatch(first.cluster_id, 0.9, 0.2, 2))
    second = engine.submit("Power cut in Ameerpet since morning")
    assert second.cluster_id == first.cluster_id
    assert second.priority > first.priority


def test_api_validates_input(client):
    assert client.post("/analyze", json={"text": ""}).status_code == 422
    assert client.post("/analyze", json={"text": "no water", "locality_id": "atlantis"}).status_code == 422
    assert client.get("/complaints?status=lost").status_code == 422
    r = client.post("/complaints/voice", files={"audio": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 422


def test_api_lists_complaints_without_location(client):
    assert client.post("/complaints", json={"text": "no water since morning"}).status_code == 200
    rows = client.get("/complaints").json()
    assert any(r["lat"] is None for r in rows)
    assert client.get("/hotspots").status_code == 200


# ------------------------------------------------------------------------------------------- chatbot
@pytest.fixture
def bot(engine):
    from awaaz.chat import ChatBot

    return ChatBot(engine)


def _talk(bot, *messages):
    from awaaz.chat import Conversation

    conv, replies = Conversation(), []
    for m in messages:
        replies = bot.respond(conv, m)
    return conv, replies


def test_chat_asks_for_location_in_telugu_then_files(bot):
    conv, replies = _talk(bot, "మా కాలనీలో మూడు రోజులుగా నీళ్లు రావడం లేదు")
    assert conv.draft is not None and "ఎక్కడ" in replies[0].text
    conv, replies = _talk(bot, "మా కాలనీలో మూడు రోజులుగా నీళ్లు రావడం లేదు", "కూకట్‌పల్లి")
    assert conv.draft is None and replies[0].ticket["locality_id"] == "kukatpally"
    assert "టికెట్" in replies[0].text


def test_chat_gives_safety_advice_for_hazards(bot):
    _, replies = _talk(bot, "A live wire has fallen on the road in Ameerpet")
    assert replies[0].ticket["urgency"] == "critical"
    assert any("1912" in r.text for r in replies)


def test_chat_gives_up_asking_for_location(bot):
    conv, replies = _talk(bot, "garbage has not been collected for a week", "xyz", "abc")
    assert conv.draft is None and replies[0].ticket["locality_id"] is None


def test_chat_lets_citizen_pick_department_when_models_disagree(bot):
    from awaaz.chat import Conversation

    conv = Conversation()
    replies = bot.respond(conv, "There is a big problem near my house in Uppal")
    if conv.draft is None:  # this backend happened to agree with itself: nothing to clarify
        pytest.skip("models agreed")
    chosen = conv.draft.options[1]
    replies = bot.respond(conv, "2")
    assert replies[0].ticket["department"] == chosen


def test_chat_tracks_tickets_and_ignores_chit_chat(bot):
    _, replies = _talk(bot, "Streetlight not working near Abids bus stop")
    ticket_id = replies[0].ticket["id"]
    conv, replies = _talk(bot, ticket_id.lower())
    assert ticket_id in replies[0].text and conv.draft is None
    conv, replies = _talk(bot, "ok")
    assert conv.draft is None and replies[0].ticket is None


def test_status_lifecycle(engine):
    tk = engine.submit("Pothole on the main road in Miyapur")
    engine.store.set_cluster_status(tk.cluster_id, "in_progress")
    row = engine.store.get(tk.id)
    assert row["status"] == "in_progress" and row["updated_at"]


def test_spikes_flag_a_sudden_cluster():
    from datetime import datetime, timedelta

    import pandas as pd

    from awaaz.cluster import spikes

    now = datetime.now()
    rows = [{"department": "water", "locality_id": "uppal", "locality": "Uppal", "urgency": "high",
             "created_at": now - timedelta(hours=h)} for h in (1, 5, 9, 20)]
    rows += [{"department": "roads", "locality_id": "abids", "locality": "Abids", "urgency": "low",
              "created_at": now - timedelta(days=d)} for d in range(1, 14)]  # steady, not a spike
    sp = spikes(pd.DataFrame(rows), now)
    assert list(sp["locality"]) == ["Uppal"]


def test_api_chat_and_ticket_lookup(client):
    r = client.post("/chat", json={"session_id": "t1", "text": "Garbage not collected for a week near Abids"}).json()
    ticket_id = r["replies"][0]["ticket_id"]
    assert ticket_id
    assert client.get(f"/complaints/{ticket_id.lower()}").json()["id"] == ticket_id
    assert client.get("/complaints/AWZ-0000-000000").status_code == 404
    assert client.get("/spikes").status_code == 200


def test_chat_asks_again_when_given_details_instead_of_a_place(bot):
    conv, replies = _talk(bot, "garbage has not been collected for a week",
                          "it is smelling very badly and dogs are spreading it")
    assert conv.draft is not None and replies[0].ticket is None  # still asking where


def test_new_report_inherits_incident_status(engine, monkeypatch):
    from awaaz import pipeline

    first = engine.submit("Sewage overflowing on the road near Kukatpally bus stop")
    engine.store.set_cluster_status(first.cluster_id, "in_progress")
    monkeypatch.setattr(pipeline, "find_duplicate", lambda *a, **k: DuplicateMatch(first.cluster_id, 0.9, 0.1, 2))
    second = engine.submit("Drainage water flowing on Kukatpally main road")
    assert engine.store.get(second.id)["status"] == "in_progress"
