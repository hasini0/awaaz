"""Citizen chatbot page: WhatsApp-style conversation, typed or by voice note, in the citizen's language."""
from __future__ import annotations

import os
import tempfile
from html import escape

import streamlit as st

from awaaz import config
from awaaz.asr import AudioError
from awaaz.chat import ChatBot, Conversation, say
from awaaz.pipeline import Awaaz

from .components import html, ticket_card

EXAMPLES = [  # one per language variant the bot understands
    ("తెలుగు", "మా కాలనీలో మూడు రోజులుగా నీళ్లు రావడం లేదు"),
    ("हिन्दी", "अमीरपेट में बिजली का तार सड़क पर गिरा हुआ है"),
    ("اردو", "ٹولی چوکی میں ایک ہفتے سے کچرا نہیں اٹھایا گیا"),
    ("English", "There is a big problem near my house in Uppal"),
    ("Tenglish", "Kukatpally bus stop daggara drainage overflow avthondi"),
    ("Hinglish", "Malakpet mein machhar bahut hain, bacchon ko bukhar hai"),
]
AVATARS = {"user": "🧑", "assistant": "🎙️"}


def _state() -> tuple[Conversation, list]:
    conv = st.session_state.setdefault("conversation", Conversation())
    history = st.session_state.setdefault("chat_history", [])  # [(role, text, ticket | None)]
    return conv, history


VOICE_LANGUAGES = {None: "Auto", "te": "తెలుగు", "hi": "हिन्दी", "ur": "اردو", "en": "English"}


def _transcribe(engine: Awaaz, audio, language: str | None) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio.getvalue())
    try:
        return engine.transcribe(f.name, language)["text"]
    except AudioError as e:
        st.toast(str(e), icon="⚠️")
        return None
    finally:
        os.unlink(f.name)


def _send(engine: Awaaz, text: str, voice: bool = False, new_complaint: bool = False):
    """Run one message through the bot. `new_complaint` drops any half-finished complaint first, so an
    example clicked while the bot waits for a location is not mistaken for the answer."""
    conv, history = _state()
    if new_complaint:
        conv.draft = None
    history.append(("user", ("🎤 " if voice else "") + text, None))
    with st.spinner("Awaaz is thinking…"):
        for reply in ChatBot(engine).respond(conv, text, channel="voice" if voice else "chat"):
            history.append(("assistant", reply.text, reply.ticket))


def _messages(history: list):
    html('<div class="awz-chat-head"><span class="awz-chat-logo">🎙️</span><div><b>Awaaz civic helpline</b>'
         '<div class="awz-chat-sub">● online · తెలుగు · हिन्दी · اردو · English</div></div></div>')
    with st.container(height=520, border=True):
        if not history:
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                st.markdown(say("greet", "en") + "\n\n" + say("greet", "te"))
        for role, text, ticket in history:
            with st.chat_message(role, avatar=AVATARS[role]):
                # the hidden marker lets the CSS right-align citizen bubbles, like a messaging app
                marker = '<span class="awz-me"></span>' if role == "user" else ""
                st.markdown(marker + escape(text).replace("\n", "  \n"), unsafe_allow_html=True)
                if ticket:
                    with st.expander("🔍 How the AI decided"):
                        ticket_card(ticket, show_map=False)


def render(engine: Awaaz):
    conv, history = _state()
    left, right = st.columns([1.6, 1], gap="large")
    with right:
        html('<div class="awz-card"><div class="awz-big">💬 Talk to Awaaz</div>'
             '<div class="awz-muted">Like a WhatsApp helpline. Type, or tap 🎤 in the message box to send a '
             'voice note. Awaaz answers in your language, asks for the location if you forgot it, and asks '
             'you to pick the department when its models disagree. Send a ticket number to track it.</div></div>')
        st.markdown("**Try a message**")
        for i, (lang, text) in enumerate(EXAMPLES):
            if st.button(f"{lang} · {text}", key=f"ex_{i}", width="stretch"):
                _send(engine, text, new_complaint=True)
                st.rerun()
        if history and st.button("↺ New conversation", width="stretch"):
            st.session_state.pop("conversation")
            st.session_state.pop("chat_history")
            st.rerun()
    with left:
        _messages(history)
        # very short voice notes can be misheard as the wrong language; picking it makes speech reliable
        voice_lang = st.segmented_control("🎤 Voice note language", list(VOICE_LANGUAGES), default=None,
                                          format_func=VOICE_LANGUAGES.get, key="voice_lang")
        if conv.draft and conv.draft.options:  # quick replies for "which department?"
            depts = config.departments()
            cols = st.columns(len(conv.draft.options))
            for i, (col, dept_id) in enumerate(zip(cols, conv.draft.options)):
                d = depts[dept_id]
                if col.button(f"{i + 1}️⃣ {d.icon} {d.name_in(conv.lang)}", key=f"quick_{i}", width="stretch"):
                    _send(engine, str(i + 1))
                    st.rerun()
        msg = st.chat_input("Type your complaint, or tap 🎤 to speak…", accept_audio=True)
        if msg:
            if msg.audio:
                text = _transcribe(engine, msg.audio, voice_lang)
                if not text:
                    history.append(("assistant", say("unheard", conv.lang), None))
                else:
                    _send(engine, text, voice=True)
            elif msg.text.strip():
                _send(engine, msg.text)
            st.rerun()
