"""Citizen tab: speak, upload or type a complaint and see the routed ticket."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from awaaz import config
from awaaz.asr import AudioError
from awaaz.pipeline import Awaaz
from awaaz.text import LANG_NAMES

from .components import html, ticket_card

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_AUDIO = ROOT / "data" / "audio_samples"
LANG_CHOICES = {"Auto-detect": None, "తెలుగు (Telugu)": "te", "हिन्दी (Hindi)": "hi", "اردو (Urdu)": "ur",
                "English": "en"}
INTRO = """<div class="awz-card"><div class="awz-big">How it works</div><ol>
<li>Speak or type in <b>any of 4 languages</b> (or a mix)</li>
<li>AI identifies the <b>department</b> and <b>urgency</b></li>
<li>Safety rules catch <b>life-threatening</b> hazards</li>
<li>The <b>location</b> is read from your words and pinned</li>
<li>Repeat reports are <b>merged</b> so officers see real hotspots</li></ol></div>"""


@st.cache_data
def _examples() -> pd.DataFrame:
    return pd.read_json(ROOT / "data" / "test_handwritten.jsonl", lines=True)


def _transcribe(engine: Awaaz, audio, lang: str | None) -> dict:
    """Recorded/uploaded audio or a bundled sample clip -> ASR result."""
    if isinstance(audio, Path):
        return engine.transcribe(str(audio), lang)
    suffix = Path(getattr(audio, "name", "a.wav")).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio.getvalue())
    try:
        return engine.transcribe(f.name, lang)
    finally:
        os.unlink(f.name)


def _inputs():
    """Returns (audio, text, language_hint, locality_id, submitted)."""
    mode = st.radio("How do you want to report?", ["🎤 Speak", "⌨️ Type", "📁 Upload audio"], horizontal=True)
    audio, text, lang = None, "", None

    if mode == "🎤 Speak":
        lang = LANG_CHOICES[st.selectbox("Language you will speak", list(LANG_CHOICES))]
        audio = st.audio_input("Tap 🎤, describe the problem and where it is, then tap ■ — it is sent automatically")
    elif mode == "📁 Upload audio":
        lang = LANG_CHOICES[st.selectbox("Language in the recording", list(LANG_CHOICES))]
        audio = st.file_uploader("Voice note (wav / mp3 / m4a / ogg)", type=["wav", "mp3", "m4a", "ogg", "webm"])
        clips = sorted(SAMPLE_AUDIO.glob("*.wav")) if SAMPLE_AUDIO.exists() else []
        pick = st.selectbox("…or use a sample voice note", ["—"] + [c.name for c in clips])
        if pick != "—":
            audio = SAMPLE_AUDIO / pick
            st.audio(str(audio))
    else:
        ex = _examples()
        labels = ["— write your own —"] + [f"[{LANG_NAMES[r.lang]}] {r.text[:70]}" for r in ex.itertuples()]
        i = st.selectbox("Try an example", range(len(labels)), format_func=lambda k: labels[k])
        text = st.text_area("Describe the problem", value="" if i == 0 else ex.text[i - 1], height=120,
                            placeholder="e.g. మా కాలనీలో మూడు రోజులుగా నీళ్లు రావడం లేదు, కూకట్‌పల్లి")

    localities = {loc["name"]: loc["id"] for loc in config.city()["localities"]}
    choice = st.selectbox("Locality (optional — AI reads it from your complaint)", ["Auto-detect", *sorted(localities)])
    submitted = st.button("Submit complaint", type="primary", width="stretch")
    return audio, text, lang, localities.get(choice), submitted


def _submit(engine: Awaaz, audio, text: str, lang: str | None, loc_id: str | None):
    st.session_state["last_audio"] = getattr(audio, "file_id", None)  # handle each recording once, even if it fails
    asr = None
    if audio is not None:
        with st.spinner("Transcribing speech…"):
            try:
                asr = _transcribe(engine, audio, lang)
            except AudioError as e:
                st.error(str(e))
                return
        text = asr["text"]
        if not text.strip():
            st.error("No speech was recognised. Please speak clearly and try again.")
            return
    if not text.strip():
        st.error("Nothing to submit yet. Recording? Tap ■ to stop it — it is submitted automatically. "
                 "Otherwise type a complaint or pick a voice note.")
        return
    with st.spinner("Understanding complaint…"):
        tk = engine.submit(text, channel="voice" if asr else "web", locality_id=loc_id)
    st.session_state["ticket"] = (tk.to_dict(), asr)


def render(engine: Awaaz):
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        audio, text, lang, loc_id, submitted = _inputs()
        # a finished recording or a new upload is processed at once; no need to also press Submit
        new_audio = getattr(audio, "file_id", None) not in (None, st.session_state.get("last_audio"))
        if submitted or new_audio:
            _submit(engine, audio, text, lang, loc_id)
    with right:
        if "ticket" in st.session_state:
            ticket_card(*st.session_state["ticket"])
        else:
            html(INTRO)
