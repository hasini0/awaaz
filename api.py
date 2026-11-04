"""REST API so any channel (WhatsApp bot, IVR, mobile app, SMS gateway) can plug into Awaaz.

    uvicorn api:app --reload
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from awaaz import config
from awaaz.asr import AudioError
from awaaz.chat import ChatBot, Conversation
from awaaz.cluster import hotspots, spikes
from awaaz.pipeline import Awaaz

app = FastAPI(title="Awaaz API", version="1.0.0",
              description="Voice-first multilingual civic grievance routing")
_engine: Awaaz | None = None
_engine_lock = threading.Lock()


def engine() -> Awaaz:
    """Models load once, on the first request (sync endpoints run in a thread pool, hence the lock)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Awaaz()
    return _engine


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe rows (timestamps as strings, NaN as null)."""
    for col in df.select_dtypes("datetime").columns:
        df = df.assign(**{col: df[col].astype(str)})
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


MAX_CONVERSATIONS = 10_000
_conversations: dict[str, Conversation] = {}  # in memory; a WhatsApp adapter would key this by phone hash


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2000)


class ComplaintIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    channel: str = "api"
    locality_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "classifier": engine().backend, "complaints": engine().store.count()}


@app.post("/analyze")
def analyze(c: ComplaintIn):
    """Classify a complaint without filing it."""
    try:
        return engine().analyze(c.text, c.locality_id).to_dict()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/complaints")
def file_complaint(c: ComplaintIn):
    try:
        return engine().submit(c.text, c.channel, c.locality_id).to_dict()
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/complaints/voice")
def file_voice(audio: UploadFile = File(...), language: str | None = Form(None), channel: str = Form("voice")):
    suffix = Path(audio.filename or "a.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio.file.read())
    try:
        asr = engine().transcribe(f.name, language)
        if not asr["text"].strip():
            raise AudioError("No speech was recognised in the recording.")
        ticket = engine().submit(asr["text"], channel)
    except ValueError as e:  # AudioError is a ValueError
        raise HTTPException(422, str(e)) from e
    finally:
        os.unlink(f.name)
    return {"transcription": asr, "ticket": ticket.to_dict()}


@app.post("/chat")
def chat(c: ChatIn):
    """One turn of the multilingual chatbot. A WhatsApp/Telegram/SMS adapter only has to forward messages here."""
    if c.session_id not in _conversations and len(_conversations) >= MAX_CONVERSATIONS:
        _conversations.pop(next(iter(_conversations)))  # forget the oldest conversation
    conv = _conversations.setdefault(c.session_id, Conversation())
    replies = ChatBot(engine()).respond(conv, c.text, channel="api-chat")
    return {"replies": [{"text": r.text, "ticket_id": r.ticket["id"] if r.ticket else None} for r in replies]}


@app.get("/complaints/{ticket_id}")
def get_complaint(ticket_id: str):
    """Ticket status lookup for citizens."""
    row = engine().store.get(ticket_id.upper())
    if not row:
        raise HTTPException(404, f"No ticket {ticket_id!r}.")
    return row


@app.get("/complaints")
def list_complaints(status: str = Query("open", pattern="^(open|in_progress|resolved)$"), department: str | None = None,
                    limit: int = Query(100, ge=1, le=1000)):
    if department and department not in config.departments():
        raise HTTPException(422, f"Unknown department {department!r}.")
    df = engine().store.df()
    df = df[df["status"] == status]
    if department:
        df = df[df["department"] == department]
    return _records(df.sort_values("priority", ascending=False).head(limit))


@app.get("/hotspots")
def get_hotspots():
    return _records(hotspots(engine().store.df()))


@app.get("/spikes")
def get_spikes():
    """Early-warning alerts: localities reporting far above their usual rate."""
    return _records(spikes(engine().store.df()))


@app.get("/departments")
def get_departments():
    return {k: vars(v) for k, v in config.departments().items()}
