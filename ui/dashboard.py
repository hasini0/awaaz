"""Officer page: early-warning spikes, live map, priority queue (duplicates merged), hotspots and trends."""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from awaaz import config
from awaaz.cluster import hotspots, spikes
from awaaz.pipeline import Awaaz
from awaaz.store import ACTIVE
from awaaz.text import LANG_NAMES

from .components import DEPTS, LEVELS, dept_label, html

URGENCY_ORDER = list(LEVELS)
STATUS_LABELS = {"open": "🆕 open", "in_progress": "🛠️ in progress", "resolved": "✅ resolved"}
CHART_LAYOUT = dict(height=330, margin=dict(l=0, r=0, t=40, b=0))


def _filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (rows matching every filter, rows matching only the department/urgency filters).
    Spike detection needs the second: it compares recent days with the weeks before them."""
    a, b, c = st.columns([2, 2, 1])
    depts = a.multiselect("Department", list(DEPTS), format_func=dept_label)
    levels = b.multiselect("Urgency", URGENCY_ORDER, format_func=lambda k: LEVELS[k].label)
    days = c.slider("Last N days", 1, 30, 21)
    if depts:
        df = df[df["department"].isin(depts)]
    if levels:
        df = df[df["urgency"].isin(levels)]
    return df[df["created_at"] >= datetime.now() - timedelta(days=days)], df


def _spike_alerts(df: pd.DataFrame):
    sp = spikes(df)
    hours = config.settings()["spikes"]["window_hours"]
    for r in sp.head(4).itertuples():
        usual = f"usually {r.per_day_usual:g}/day" if r.per_day_usual else "usually none"
        crit = f" · <b>{r.critical} critical</b>" if r.critical else ""
        html(f'<div class="awz-spike">📈 <b>Spike:</b> {DEPTS[r.department].icon} {DEPTS[r.department].name} in '
             f'<b>{escape(r.locality)}</b> — {r.recent} reports in the last {hours} h ({usual}){crit}</div>')


def _kpis(active: pd.DataFrame, q: pd.DataFrame, hs: pd.DataFrame):
    k = st.columns(6)
    k[0].metric("📋 Incidents", len(q), help="Open incidents; duplicate reports are merged into one", border=True)
    k[1].metric("🔴 Critical", int((q["urgency"] == "critical").sum()) if len(q) else 0, border=True)
    k[2].metric("⏰ Overdue", int(q["sla"].str.startswith("⚠️").sum()) if len(q) else 0,
                help="Past the department's resolution target (SLA)", border=True)
    k[3].metric("🔁 Merged", int(len(active) - len(q)),
                help="Citizen reports folded into an existing incident: officer visits saved", border=True)
    k[4].metric("🔥 Hotspots", len(hs), border=True)
    k[5].metric("🗣️ Languages", active["lang"].nunique(), border=True)


def _map(active: pd.DataFrame, hs: pd.DataFrame):
    city = config.city()
    m = folium.Map(location=city["center"], zoom_start=city["zoom"])
    for h in hs.itertuples():
        folium.Circle([h.lat, h.lon], radius=350 + 60 * h.reports, color=LEVELS[h.max_urgency].color,
                      fill=True, fill_opacity=0.18, weight=2,
                      tooltip=f"🔥 {DEPTS[h.department].name} — {h.reports} reports ({h.localities})").add_to(m)
    for r in active.dropna(subset=["lat"]).itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=4 + LEVELS[r.urgency].weight, color=LEVELS[r.urgency].color,
                            fill=True, fill_opacity=0.85, weight=1,
                            tooltip=f"{DEPTS[r.department].icon} {r.locality} · {LEVELS[r.urgency].label}",
                            popup=folium.Popup(f"<b>{r.id}</b><br>{escape(r.text[:160])}", max_width=280)).add_to(m)
    st_folium(m, height=430, use_container_width=True, returned_objects=[], key="dash_map")


def _most_common(s: pd.Series):
    """The locality most reports in an incident name (the first report may come from a neighbouring area)."""
    return s.mode().iat[0] if s.notna().any() else None


def priority_queue(active: pd.DataFrame) -> pd.DataFrame:
    """One row per incident (duplicate cluster), most urgent first."""
    if active.empty:
        return pd.DataFrame(columns=["priority", "urg", "dept", "locality", "reports", "status", "sla", "age_h",
                                     "sample", "langs", "cluster_id", "urgency"])
    q = (active.sort_values("created_at")
         .groupby("cluster_id")
         .agg(department=("department", "first"), locality=("locality", _most_common),
              urgency=("urgency", lambda s: max(s, key=URGENCY_ORDER.index)),
              reports=("id", "count"), priority=("priority", "max"), status=("status", "last"),
              langs=("lang", lambda s: ", ".join(sorted(set(s)))),
              first=("created_at", "min"), sample=("text", "first"))
         .sort_values(["priority", "reports"], ascending=False)
         .reset_index())
    q["age_h"] = ((datetime.now() - q["first"]).dt.total_seconds() / 3600).round(0)
    q["sla"] = ["⚠️ breached" if age > DEPTS[d].sla_hours else "ok" for age, d in zip(q["age_h"], q["department"])]
    q["dept"] = q["department"].map(dept_label)
    q["urg"] = q["urgency"].map(lambda k: LEVELS[k].label)
    q["status"] = q["status"].map(STATUS_LABELS)
    q["locality"] = q["locality"].fillna("📍 unknown")
    return q[["priority", "urg", "dept", "locality", "reports", "status", "sla", "age_h", "sample", "langs",
              "cluster_id", "urgency"]]


def _queue(engine: Awaaz, q: pd.DataFrame):
    if q.empty:
        st.info("No open incidents for this filter.")
        return
    sel = st.dataframe(q.drop(columns="urgency"), hide_index=True, width="stretch", height=360, on_select="rerun",
                       selection_mode="multi-row",
                       column_config={
                           "priority": st.column_config.ProgressColumn("priority", min_value=0, max_value=100,
                                                                       format="%.0f", width="small"),
                           "urg": st.column_config.TextColumn("urgency", width="small"),
                           "dept": st.column_config.TextColumn("department", width="medium"),
                           "reports": st.column_config.NumberColumn("reports", width="small"),
                           "age_h": st.column_config.NumberColumn("age (h)", width="small"),
                           "cluster_id": "ticket", "langs": "languages",
                           "sample": st.column_config.TextColumn("complaint", width="large")})
    rows = [i for i in sel.selection.rows if i < len(q)]  # a stale selection can outlive a filter change
    chosen = q.iloc[rows]["cluster_id"].tolist()
    a, b, c, _ = st.columns([1, 1, 1, 2])
    if a.button(f"🛠️ Start work ({len(chosen)})", disabled=not chosen, width="stretch"):
        _set_status(engine, chosen, "in_progress")
    if b.button(f"✅ Resolve ({len(chosen)})", disabled=not chosen, width="stretch"):
        _set_status(engine, chosen, "resolved")
    c.download_button("⬇️ Export CSV", q.drop(columns="urgency").to_csv(index=False).encode("utf-8"),
                      file_name=f"awaaz_queue_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv", width="stretch")


def _set_status(engine: Awaaz, cluster_ids: list[str], status: str):
    for cid in cluster_ids:
        engine.store.set_cluster_status(cid, status)
    # shown after the rerun (a toast raised right before st.rerun() is lost)
    st.session_state["dashboard_flash"] = (f"{len(cluster_ids)} incident(s) → {STATUS_LABELS[status]}. "
                                           "Citizens see the new status when they track their ticket.")
    st.rerun()


def _charts(df: pd.DataFrame):
    if df.empty:
        return
    a, b, c = st.columns(3)
    by = df.groupby(["department", "urgency"]).size().reset_index(name="n")
    by["department"] = by["department"].map(lambda k: DEPTS[k].name)
    fig = px.bar(by, x="n", y="department", color="urgency", orientation="h", title="Complaints by department",
                 color_discrete_map={k: v.color for k, v in LEVELS.items()},
                 category_orders={"urgency": URGENCY_ORDER})
    a.plotly_chart(fig.update_layout(**CHART_LAYOUT, yaxis_title=None, xaxis_title=None), width="stretch")

    langs = df["lang"].map(LANG_NAMES).value_counts().reset_index()
    fig = px.pie(langs, names="lang", values="count", hole=.55, title="Reporting language")
    b.plotly_chart(fig.update_layout(**CHART_LAYOUT), width="stretch")

    daily = df.set_index("created_at").resample("D").size().reset_index(name="n")
    fig = px.area(daily, x="created_at", y="n", title="Daily complaints")
    c.plotly_chart(fig.update_layout(**CHART_LAYOUT, xaxis_title=None, yaxis_title=None), width="stretch")


def _hotspot_table(hs: pd.DataFrame):
    st.markdown("##### 🔥 Hotspots")
    if hs.empty:
        st.caption("No clusters of unresolved complaints right now.")
        return
    st.caption("Clusters of unresolved complaints found by DBSCAN (circles on the map).")
    t = hs.assign(department=hs["department"].map(dept_label),
                  max_urgency=hs["max_urgency"].map(lambda k: LEVELS[k].label))
    st.dataframe(t[["department", "localities", "reports", "max_urgency"]], hide_index=True, width="stretch",
                 height=390, column_config={"max_urgency": "worst urgency"})


def render(engine: Awaaz):
    if flash := st.session_state.pop("dashboard_flash", None):
        st.toast(flash, icon="✅")
    view, history = _filters(engine.store.df())
    active = view[view["status"].isin(ACTIVE)]
    hs = hotspots(view)
    q = priority_queue(active)
    _spike_alerts(history)
    _kpis(active, q, hs)

    st.markdown("##### 🚦 Priority queue — one row per incident, most urgent first")
    _queue(engine, q)

    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        st.markdown("##### 🗺️ Live complaint map")
        _map(active, hs)
    with right:
        _hotspot_table(hs)
    _charts(view)
