"""'How it works' page: architecture diagram and extension points."""
import streamlit as st

ARCHITECTURE = """
digraph G {
  rankdir=TB; nodesep=0.35; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica", fontsize=17, margin="0.15,0.08"];
  edge [color="#64748b"];
  subgraph cluster_in { label="Channels (pluggable)"; style=dashed; color="#94a3b8";
    chat [label="💬 Web chat"]; mic [label="🎤 Voice note"]; wa [label="WhatsApp*"]; ivr [label="IVR call*"]; }
  asr   [label="Language ID (te/hi/ur/en)\\n→ per-language Whisper\\n(Telugu-tuned for te)"];
  dlg   [label="Chatbot dialogue\\nasks for missing location\\nor department · replies\\nin citizen's language",
         fillcolor="#e0f2fe"];
  clf   [label="MuRIL classifier\\n→ department + urgency"];
  com   [label="Second opinions\\nTF-IDF + e5\\n(disagree → ask citizen)", fillcolor="#fef9c3"];
  rules [label="Safety rules + advice\\n(can only raise urgency)", fillcolor="#fee2e2"];
  loc   [label="Gazetteer locator\\n4 scripts · fuzzy · phonetic"];
  dedup [label="Cross-lingual duplicate\\nmatching (e5 + 1.5 km)"];
  prio  [label="Priority score\\n+ SLA"];
  db    [label="Ticket store\\nopen → in progress → resolved", shape=cylinder, fillcolor="#f1f5f9"];
  dash  [label="🏛️ Officer dashboard\\nspike alerts · map · queue\\nDBSCAN hotspots", fillcolor="#dcfce7"];
  mic -> asr; wa -> asr; ivr -> asr; asr -> dlg; chat -> dlg;
  dlg -> clf -> com -> rules -> prio; dlg -> loc -> dedup -> prio -> db -> dash;
  {rank=same; clf; loc;} {rank=same; com; dedup;}
  db -> dlg [label=" ticket status", style=dashed, fontsize=9];
}"""

EXTENDING = """
**Built to extend.** Every stage is a separate module behind a config file:

| Want to… | Change |
|---|---|
| Launch in another city | add `config/cities/<city>.yaml` (locality gazetteer), set `city:` in `config/settings.yaml` |
| Add a department | add it (with local names) to `config/departments.yaml` + training phrases, retrain |
| Reply in another language | add a block to `config/messages.yaml` |
| Better speech recognition for a language | `asr.overrides.<lang>` → any Hugging Face Whisper model |
| Swap the classifier | `classifier.backend: tfidf / embed / ensemble / muril` |
| WhatsApp / Telegram / SMS bot | forward each message to `POST /chat` and send back the replies |
| IVR phone line | send the recording to `POST /complaints/voice` |
| Tune early-warning alerts | `spikes:` in `config/settings.yaml` |
"""


def render():
    diagram, extend = st.columns([1, 1.15], gap="large")
    with diagram:
        st.graphviz_chart(ARCHITECTURE, width="stretch")
    with extend:
        st.markdown(EXTENDING)
    st.caption("* WhatsApp / IVR adapters are on the roadmap; they call the same REST API (`api.py`) as the web app.")
