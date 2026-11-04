"""Model-performance tab, rendered from reports/metrics.json written by the training scripts."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from awaaz import config
from awaaz.classify import best_backend
from awaaz.text import LANG_NAMES


@st.cache_data
def load_metrics() -> dict | None:
    p = config.REPORTS_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def _summary_table(models: dict) -> pd.DataFrame:
    rows = []
    for m in models.values():
        h = m["handwritten"]
        rows.append({"model": m["display"],
                     "dept macro-F1": h["dept_macro_f1"], "dept accuracy": h["dept_accuracy"],
                     "urgency macro-F1": h["urg_macro_f1"], "urgency ±1 level": h["urg_within_one_level"],
                     "critical recall": h["critical_recall"], "under-triage": h["under_triage_rate"],
                     "unseen-phrasing dept-F1": m["synthetic_unseen_phrasing"]["dept_macro_f1"],
                     "latency (ms)": m.get("latency_ms_per_complaint")})
    return pd.DataFrame(rows)


def _per_language_chart(models: dict):
    rows = [{"model": m["display"], "language": LANG_NAMES.get(lang, lang), "dept macro-F1": s["dept_macro_f1"]}
            for m in models.values() for lang, s in m["handwritten"]["per_language"].items()]
    fig = px.bar(pd.DataFrame(rows), x="language", y="dept macro-F1", color="model", barmode="group",
                 title="Department macro-F1 per language (hand-written set)")
    st.plotly_chart(fig.update_layout(height=360, yaxis_range=[0, 1]), width="stretch")


def _safety(models: dict):
    """Reported for the model the app actually runs (chosen on the unseen-phrasing split, not this test set)."""
    st.markdown("#### 🛟 Safety layer")
    deployed = models.get(best_backend()) or next(iter(models.values()))
    h = deployed["handwritten"]
    st.markdown(f"Deployed model: *{deployed['display']}*\n"
                f"- Urgency macro-F1, model alone **{h['urg_macro_f1_model_only']:.3f}** → with safety rules "
                f"**{h['urg_macro_f1']:.3f}**\n"
                f"- Critical-hazard recall **{h['critical_recall']:.0%}** (precision {h['critical_precision']:.0%})\n"
                f"- Under-triage rate (urgency predicted too low) **{h['under_triage_rate']:.1%}**")


def _voice(v: dict | None):
    if not v:
        return
    te_model = (v.get("asr_overrides") or {}).get("te", {}).get("model", "—")
    st.markdown(f"#### 🎤 Voice, end to end (no language hint)\n"
                f"- {v['clips']} clips in Telugu, Hindi and English: language identified "
                f"**{v.get('language_id_accuracy', 0):.0%}**, department **{v['department_accuracy']:.0%}**, "
                f"locality **{v['locality_accuracy']:.0%}**\n"
                f"- Telugu goes to a Telugu-tuned Whisper (`{te_model}`); "
                f"base Whisper-small wrote Telugu speech in Devanagari\n"
                f"- Test clips are synthetic text-to-speech; real recordings are the next step")


def _dedup(dd: dict):
    st.markdown("#### 🔁 Cross-language duplicate detection")
    if not dd:
        return
    full, text = dd["text_plus_geo"], dd["text_only"]
    st.markdown(f"- Cosine threshold (calibrated on synthetic cross-lingual pairs): **{dd['threshold']}**\n"
                f"- Meaning + distance (as deployed): precision **{full['precision']:.2f}**, recall "
                f"**{full['recall']:.2f}**, F1 **{full['f1']:.2f}** on {dd['pairs']} hand-written pairs\n"
                f"- Meaning alone: F1 {text['f1']:.2f} — geography removes most false merges\n"
                f"- Same incident reported in a *different language*: recall **{dd['cross_language_recall']:.0%}**")


def _headline(mt: dict, models: dict):
    """The four numbers that matter, as cards."""
    deployed = models.get(best_backend()) or next(iter(models.values()))
    h = deployed["handwritten"]
    cards = []
    if com := mt.get("committee"):
        cards.append(("🎯 Routing accuracy", f"{com['routing_accuracy_with_one_tap']:.0%}",
                      f"with one tap when models disagree ({com['main_model_accuracy']:.0%} model alone)"))
    cards.append(("🛟 Critical hazards caught", f"{h['critical_recall']:.0%}",
                  f"precision {h['critical_precision']:.0%} · rules only raise urgency"))
    if v := mt.get("voice"):
        cards.append(("🎤 Voice, auto language", f"{round(v['department_accuracy'] * v['clips'])}/{v['clips']}",
                      f"clips routed and located · language ID {v.get('language_id_accuracy', 0):.0%}"))
    if dd := mt.get("dedup"):
        cards.append(("🔁 Cross-language merges", f"{dd['cross_language_recall']:.0%}",
                      f"same incident, different language · precision {dd['text_plus_geo']['precision']:.0%}"))
    for col, (title, value, note) in zip(st.columns(len(cards)), cards):
        with col.container(border=True):
            st.markdown(f'<div class="awz-muted">{title}</div><div class="awz-kpi">{value}</div>'
                        f'<div class="awz-muted">{note}</div>', unsafe_allow_html=True)


def _committee(com: dict | None):
    if not com:
        return
    st.markdown(f"#### 🤝 Ask the citizen when unsure\n"
                f"Fine-tuned models are confidently wrong on vague complaints (\"big problem near my house\"). "
                f"Two fast models ({', '.join(com['second_opinions'])}) give a second opinion; when they disagree "
                f"the chatbot offers two departments instead of guessing.\n"
                f"- Asked on **{com['asked_rate']:.0%}** of complaints · accuracy when not asked "
                f"**{com['accuracy_when_confident']:.0%}**\n"
                f"- When asked, the right department is one of the two options "
                f"**{com['correct_option_offered_when_asked']:.0%}** of the time\n"
                f"- Routing accuracy: **{com['main_model_accuracy']:.1%} → "
                f"{com['routing_accuracy_with_one_tap']:.1%}**")


def render():
    mt = load_metrics()
    if not mt:
        st.info("Run `python scripts/train.py` to generate the evaluation report.")
        return
    models = mt["models"]
    _headline(mt, models)
    st.markdown("#### Headline results — 120 hand-written complaints the models never saw")
    st.caption("Written independently of the training templates. The same 25 incidents are described in six "
               "language variants, so per-language scores are directly comparable.")
    st.dataframe(_summary_table(models).style.format(precision=3), hide_index=True, width="stretch")
    _per_language_chart(models)

    a, b = st.columns(2)
    with a:
        _safety(models)
        _committee(mt.get("committee"))
    with b:
        _dedup(mt.get("dedup", {}))
        _voice(mt.get("voice"))

    cal = models.get("muril", {}).get("calibration")
    if cal:
        st.markdown(f"#### 🎯 Calibrated confidence (temperature scaling)\n"
                    f"Fine-tuned MuRIL was accurate but under-confident. One temperature per head, fitted on the "
                    f"validation split, makes the confidence shown to officers trustworthy: expected calibration "
                    f"error **{cal['ece_dept_before']:.2f} → {cal['ece_dept_after']:.2f}** (department) and "
                    f"**{cal['ece_urg_before']:.2f} → {cal['ece_urg_after']:.2f}** (urgency), accuracy unchanged.")

    figs = sorted((config.REPORTS_DIR / "figures").glob("cm_*.png"))
    if figs:
        st.markdown("#### Confusion matrices (hand-written set)")
        cols = st.columns(2)
        for i, f in enumerate(figs):
            cols[i % 2].image(str(f), width="stretch")
    d = mt["data"]
    st.caption(f"Training: {d['train']} synthetic complaints · validation {d['val']} · "
               f"unseen-phrasing test {d['test_synthetic']} · hand-written test {d['test_handwritten']}.")
