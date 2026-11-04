"""Awaaz web app.

    streamlit run app.py
"""
import threading
from pathlib import Path

import streamlit as st

from awaaz import asr, config
from awaaz.chat import TICKET_ID, ChatBot, Conversation
from awaaz.classify import available_backends
from awaaz.demo import seed
from awaaz.pipeline import Awaaz
from awaaz.store import Store
from ui import about, chat, dashboard, model_report, report
from ui.components import CSS, html

st.set_page_config(page_title="Awaaz · Civic Voice AI", page_icon="🎙️", layout="wide")
html(CSS)


@st.cache_resource(show_spinner="Loading AI models (first run downloads them)…")
def get_engine() -> Awaaz:
    engine = Awaaz(Store())
    if engine.store.count() == 0:
        seed(engine)
    return engine


@st.cache_resource
def preload_speech_models():
    """Speech models load in the background while people read or type, instead of on the first voice note.
    Skipped on Streamlit Community Cloud (apps live under /mount/src), whose free memory can't hold them."""
    if not Path("/mount/src").exists():
        threading.Thread(target=asr.preload, daemon=True).start()


def hero(engine: Awaaz) -> str:
    stats = ["🗣️ 4 languages · 6 writing styles", f"🏛️ {len(config.departments())} departments",
             f"📍 {len(config.city()['localities'])} localities", f"🧠 {engine.backend} classifier",
             "🛟 safety rules can only raise urgency"]
    return ('<div class="awz-hero"><h1>🎙️ Awaaz <span>· ఆవాజ్ · आवाज़ · آواز</span></h1>'
            '<p>A civic helpline anyone can use: speak or type a complaint in Telugu, Hindi, Urdu or English. '
            'Awaaz routes it to the right department, flags danger, replies in your language, and merges repeat '
            'reports so officers see real hotspots.</p>'
            + "".join(f'<span class="awz-stat">{s}</span>' for s in stats) + "</div>")


def sidebar(engine: Awaaz):
    city = config.city()
    with st.sidebar:
        st.markdown("### ⚙️ System")
        st.markdown(f"**Classifier:** `{engine.backend}`  \n**Available:** {', '.join(available_backends())}  \n"
                    f"**City:** {city['city']} ({len(city['localities'])} localities)  \n"
                    f"**Departments:** {len(config.departments())}  \n**Complaints in DB:** {engine.store.count()}")
        st.divider()
        st.markdown("**Languages**  \nతెలుగు · हिन्दी · اردو · English  \nplus romanised Telugu & Hindi")
        st.divider()
        ticket = st.text_input("🔎 Track a ticket", placeholder="AWZ-2611-1A2B3C").strip()
        if ticket:
            if TICKET_ID.fullmatch(ticket):
                st.info(ChatBot(engine).respond(Conversation(), ticket)[0].text.replace("\n", "  \n"))
            else:
                st.caption("Ticket numbers look like AWZ-2611-1A2B3C.")
        st.divider()
        if st.button("↺ Reset demo data", width="stretch"):
            with st.spinner("Re-seeding…"):
                seed(engine)
            for key in ("ticket", "conversation", "chat_history"):
                st.session_state.pop(key, None)
            st.rerun()
        st.caption("⚠️ Prototype. Urgency is decision *support*: safety rules can only raise it, and "
                   "low-confidence tickets go to human review.")


engine = get_engine()
preload_speech_models()
html(hero(engine))
sidebar(engine)

# Only the selected page is rendered (st.tabs would render maps inside hidden tabs at zero width).
# ?page=dashboard etc. gives each page a shareable link.
PAGES = {
    "chat": ("💬 Chat with Awaaz", lambda: chat.render(engine)),
    "report": ("🎤 Report form", lambda: report.render(engine)),
    "dashboard": ("🏛️ Officer dashboard", lambda: dashboard.render(engine)),
    "model": ("📊 Model performance", model_report.render),
    "about": ("🧩 How it works", about.render),
}
start = st.query_params.get("page", "chat")
page = st.segmented_control("Page", list(PAGES), format_func=lambda k: PAGES[k][0],
                            default=start if start in PAGES else "chat", label_visibility="collapsed")
st.write("")
PAGES[page or "chat"][1]()
