"""Shared look-and-feel: CSS, badges and the ticket card."""
from __future__ import annotations

from html import escape

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from awaaz import config
from awaaz.text import LANG_NAMES

DEPTS = config.departments()
LEVELS = {u.id: u for u in config.urgency_levels()}

CSS = """
<style>
.block-container {padding-top: 3.4rem; max-width: 1400px;}
.awz-hero {background: linear-gradient(120deg, #4c1d95 0%, #7c3aed 55%, #db2777 100%); color: white;
           border-radius: 18px; padding: 1rem 1.5rem .9rem; margin-bottom: 1.2rem;}
.awz-hero h1 {margin: 0; padding: 0; font-size: 1.9rem; color: white;}
.awz-hero h1 span {font-weight: 400; opacity: .85;}
.awz-hero p {margin: .35rem 0 .8rem; font-size: 1.02rem; opacity: .92; max-width: 60rem;}
.awz-stat {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
           border-radius: 999px; padding: .22rem .8rem; margin: .15rem .35rem .15rem 0; font-size: .86rem;}
.awz-card {border: 1px solid rgba(100,116,139,.25); border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: .8rem;}
.awz-badge {display:inline-block; padding: .18rem .65rem; border-radius: 999px; color: white; font-weight: 600;
            font-size: .85rem; margin-right: .35rem;}
.awz-chip {display:inline-block; padding: .12rem .55rem; border-radius: 6px; background: rgba(100,116,139,.14);
           font-size: .85rem; margin: .12rem .25rem .12rem 0;}
.awz-big {font-size: 1.45rem; font-weight: 700; margin: .1rem 0 .2rem 0;}
.awz-kpi {font-size: 2.1rem; font-weight: 800; line-height: 1.15; color: #7c3aed;}
.awz-muted {color: #64748b; font-size: .88rem;}
.awz-alert {border-left: 4px solid #dc2626; background: rgba(220,38,38,.08); padding: .6rem .8rem;
            border-radius: 6px; margin: .5rem 0;}
.awz-spike {border-left: 4px solid #f97316; background: rgba(249,115,22,.09); padding: .55rem .8rem;
            border-radius: 6px; margin: .35rem 0;}
.awz-chat-head {display: flex; gap: .7rem; align-items: center; background: #075e54; color: white;
                 border-radius: 12px 12px 0 0; padding: .55rem .9rem; margin-bottom: -.6rem;}
.awz-chat-logo {font-size: 1.5rem; background: rgba(255,255,255,.15); border-radius: 50%; width: 2.3rem;
                height: 2.3rem; display: flex; align-items: center; justify-content: center;}
.awz-chat-sub {font-size: .78rem; opacity: .85;}
[data-testid="stChatMessage"] {background: rgba(100,116,139,.08); border-radius: 14px; margin-right: 12%;}
[data-testid="stChatMessage"]:has(.awz-me) {background: #dcf8c6; margin-left: 15%; margin-right: .3rem;
                                            flex-direction: row-reverse; text-align: right;}
[data-testid="stChatMessage"]:has(.awz-me) p {color: #111b21;}
.awz-dup {border-left: 4px solid #7c3aed; background: rgba(124,58,237,.08); padding: .6rem .8rem;
          border-radius: 6px; margin: .5rem 0;}
</style>
"""


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def badge(level_id: str) -> str:
    lv = LEVELS[level_id]
    return f'<span class="awz-badge" style="background:{lv.color}">{lv.label}</span>'


def dept_label(dept_id: str) -> str:
    d = DEPTS[dept_id]
    return f"{d.icon} {d.name}"


def ticket_card(tk: dict, asr: dict | None = None, show_map: bool = True):
    """Everything a citizen (and a judge) needs to see about one processed complaint.
    `show_map=False` inside expanders: a folium map rendered while hidden gets zero width."""
    d = DEPTS[tk["department"]]
    if asr:
        lang = LANG_NAMES.get(asr["language"], asr["language"] or "?")
        if asr.get("language_prob") is not None:
            lang += f' detected, {asr["language_prob"]:.0%}'
        html(f'<div class="awz-card"><div class="awz-muted">🎧 Heard · {lang} · {asr["backend"]} · '
             f'{asr["latency_s"]:.1f}s</div><div style="font-size:1.1rem">“{escape(tk["text"])}”</div></div>')

    filed = f'Ticket <b>{tk["id"]}</b> filed' if tk.get("id") else "Preview (not filed)"
    dept_source = ("department chosen by the citizen" if tk.get("department_by_citizen")
                   else f"department confidence {tk['dept_conf']:.0%}")
    html(f"""
<div class="awz-card">
  <div class="awz-muted">{filed} · language detected: <b>{tk['lang_name']}</b></div>
  <div class="awz-big">{d.icon} {d.name}</div>
  <div>Routed to <b>{d.agency}</b> · resolution target <b>{d.sla_hours} h</b></div>
  <div style="margin-top:.6rem">{badge(tk['urgency'])}
    <span class="awz-muted">{dept_source} · urgency confidence {tk['urg_conf']:.0%}</span>
  </div>
</div>""")

    levels = list(LEVELS)
    for h in tk["rule_hits"]:
        if levels.index(h["level"]) > levels.index(tk["model_urgency"]):
            effect = f'urgency raised to <b>{h["level"]}</b>. Model alone said <i>{tk["model_urgency"]}</i>.'
        else:
            effect = f'confirms the model\'s <b>{tk["model_urgency"]}</b> urgency.'
        html(f'<div class="awz-alert">🚨 <b>Safety rule:</b> {h["reason"]} (matched “{escape(h["keyword"])}”) → '
             f'{effect}</div>')
    if tk.get("reports", 1) > 1:
        html(f'<div class="awz-dup">🔁 <b>{tk["reports"]} citizens</b> have reported this same problem nearby '
             f'(semantic match {tk["duplicate_similarity"]:.2f}). Merged into <b>{tk["cluster_id"]}</b> — '
             f'priority now <b>{tk["priority"]:.0f}</b>/100.</div>')
    if tk.get("routing_unsure"):
        others = ", ".join(f"{k}: {DEPTS[v].name}" for k, v in tk.get("second_opinions", {}).items())
        st.warning(f"The models disagree on the department ({others}) — the citizen is asked to confirm, "
                   "and an officer reviews the routing.")
    elif tk["needs_review"]:
        st.warning("No location found — flagged for an officer to verify.")

    why, where = st.columns(2)
    with why:
        st.markdown("**Why this department?**")
        html("".join(f'<span class="awz-chip">{escape(k)}</span>' for k in tk["keywords"]) or "<i>—</i>")
        for dept_id, p in pd.Series(tk["dept_probs"]).nlargest(3).items():
            st.progress(float(p), text=f"{dept_label(dept_id)} — {p:.0%}")
    with where:
        st.markdown("**Location**")
        if tk["locality"]:
            html(f"📍 {escape(tk['locality'])} <span class='awz-muted'>({tk['location_method']} match)</span>")
            if show_map:
                m = folium.Map(location=[tk["lat"], tk["lon"]], zoom_start=14)
                color = "red" if tk["urgency"] in ("critical", "high") else "blue"
                folium.Marker([tk["lat"], tk["lon"]], tooltip=d.name, icon=folium.Icon(color=color)).add_to(m)
                st_folium(m, height=210, use_container_width=True, returned_objects=[], key=f"mini_{tk.get('id')}")
        else:
            st.markdown("📍 No locality found — flagged for an officer to locate.")
    st.caption("Pipeline latency: " + " · ".join(f"{k} {v:.0f} ms" for k, v in tk["timings_ms"].items()))
