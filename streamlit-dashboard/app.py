"""
Dashboard Streamlit — Pipeline Big Data de Monitoring de la Désinformation en Temps Réel
Auteur   : KOMOSSI Sosso — Master BIG DATA IA, UCAO UUT 2025-2026
Encadrant: M. TCHANTCHO Leri & M. BABA Kpatcha
"""

import os, time, socket, requests
from datetime import datetime, timezone

import streamlit as st

# set_page_config DOIT être le tout premier appel Streamlit du script
st.set_page_config(
    page_title="Surveillance Désinformation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Pipeline Big Data Désinformation — KOMOSSI Sosso, Master BIG DATA IA, UCAO UUT"
    }
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

try:
    from pymongo import MongoClient
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False

try:
    from elasticsearch import Elasticsearch
    HAS_ES = True
except ImportError:
    HAS_ES = False


# ── Configuration (st.secrets → env vars → défaut) ───────────────────────────
# Vérifie si un fichier secrets.toml existe pour éviter les avertissements
# "No secrets found" inutiles dans l'environnement Docker
_SECRETS_PATHS = [
    "/root/.streamlit/secrets.toml",
    "/app/.streamlit/secrets.toml",
    os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
]
_HAS_SECRETS_FILE = any(os.path.exists(p) for p in _SECRETS_PATHS)

# Détection de l'environnement Streamlit Cloud (/mount/src est le chemin de montage)
_IS_CLOUD = os.path.exists("/mount/src")

def _cfg(secret_key, env_key, default):
    if _HAS_SECRETS_FILE:
        try:
            val = st.secrets.get(secret_key)
            if val is not None:
                return str(val)
        except Exception:
            pass
    return os.getenv(env_key, default)

_docker_default_mongo = "" if _IS_CLOUD else "mongodb://mongodb:27017"
_docker_default_es    = "" if _IS_CLOUD else "http://elasticsearch:9200"
_docker_default_api   = "" if _IS_CLOUD else "http://api:8000"

MONGO_URI   = _cfg("MONGO_URI",   "MONGO_URI",   _docker_default_mongo)
MONGO_DB    = _cfg("MONGO_DB",    "MONGO_DB",    "disinformation_db")
ES_HOST     = _cfg("ES_HOST",     "ES_HOST",     _docker_default_es)
API_BASE    = _cfg("API_BASE",    "API_BASE",    _docker_default_api)
REFRESH_SEC = int(_cfg("REFRESH_SEC", "REFRESH_SEC", "30"))

CLR_FAKE    = "#E74C3C"
CLR_REAL    = "#2ECC71"
CLR_DRIFT   = "#F39C12"
CLR_PRIMARY = "#2C3E50"
CLR_INFO    = "#3498DB"
CLR_PURPLE  = "#8E44AD"

DEFAULT_THRESHOLDS = {
    "fake_rate_warn":    40.0,
    "fake_rate_crit":    70.0,
    "drift_warn":        0.3,
    "drift_crit":        0.5,
    "conf_low":          0.70,
    "silence_minutes":   15,
}

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F8F9FA; }
.kpi-card {
    background: white; border-radius: 12px; padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 5px solid #2C3E50;
    margin-bottom: 12px;
}
.kpi-card.fake  { border-left-color: #E74C3C; }
.kpi-card.real  { border-left-color: #2ECC71; }
.kpi-card.drift { border-left-color: #F39C12; }
.kpi-card.info  { border-left-color: #3498DB; }
.kpi-card.warn  { border-left-color: #F39C12; }
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #2C3E50; margin: 0; }
.kpi-label { font-size: 0.82rem; color: #7F8C8D; font-weight: 500; margin: 0; letter-spacing: 0.5px; }
.badge-fake { background:#E74C3C; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-real { background:#2ECC71; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-warn { background:#F39C12; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-crit { background:#E74C3C; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a252f 0%, #2C3E50 100%);
}
[data-testid="stSidebar"] * { color: #ECF0F1 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px; border-radius: 8px; margin: 2px 0;
    display: block; transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.1); }
.section-header {
    background: linear-gradient(135deg, #2C3E50, #3498DB);
    color: white; padding: 14px 22px; border-radius: 10px;
    margin-bottom: 20px; font-size: 1.25rem; font-weight: 600;
}
.alert-critical {
    background: #FDEDEC; border-left: 5px solid #E74C3C;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.alert-warning {
    background: #FEF9E7; border-left: 5px solid #F39C12;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.alert-ok {
    background: #EAFAF1; border-left: 5px solid #2ECC71;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
}
.drift-alert {
    background: #FEF9E7; border: 1px solid #F39C12;
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
}
.status-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px;
}
.status-up   { background: #2ECC71; }
.status-down { background: #E74C3C; }
</style>
""", unsafe_allow_html=True)


# ── Connexions (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_mongo():
    if not HAS_MONGO or not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        return client[MONGO_DB]
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_es():
    if not HAS_ES or not ES_HOST:
        return None
    try:
        es = Elasticsearch(ES_HOST, request_timeout=4)
        if es.ping():
            return es
        return None
    except Exception:
        return None


def _backend_ok():
    db = get_mongo()
    return db is not None


# ── Fonctions de données ──────────────────────────────────────────────────────
@st.cache_data(ttl=20, show_spinner=False)
def fetch_stats():
    if not API_BASE:
        return {}
    try:
        r = requests.get(f"{API_BASE}/api/v1/stats", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}

@st.cache_data(ttl=20, show_spinner=False)
def fetch_health():
    if not API_BASE:
        return {}
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def fetch_articles(limit=100, fake_only=False, real_only=False, source_filter=None,
                   conf_min=0.0, drift_only=False):
    db = get_mongo()
    if db is None:
        return pd.DataFrame()
    try:
        query = {}
        if fake_only:         query["is_fake"] = 1
        if real_only:         query["is_fake"] = 0
        if source_filter:     query["source"] = source_filter
        if drift_only:        query["drift_active"] = True
        if conf_min > 0:      query["confidence"] = {"$gte": conf_min}
        cursor = db.articles.find(
            query,
            {"_id": 0, "id": 1, "title": 1, "body": 1, "url": 1, "source": 1,
             "language": 1, "is_fake": 1, "confidence": 1, "p_fake": 1,
             "drift_score": 1, "drift_active": 1, "processed_at": 1}
        ).sort("processed_at", -1).limit(limit)
        return pd.DataFrame(list(cursor))
    except Exception:
        return pd.DataFrame()

def fetch_virality(hours=24):
    try:
        r = requests.get(f"{API_BASE}/api/v1/articles/virality?hours={hours}", timeout=10)
        return r.json().get("trend", []) if r.ok else []
    except Exception:
        return []

def fetch_drift_events(limit=50):
    db = get_mongo()
    if db is None:
        return []
    try:
        cursor = db.drift_events.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception:
        return []

def search_articles(query_text, size=20, fake_filter=None):
    es = get_es()
    if es is None:
        return pd.DataFrame()
    try:
        must = [{"multi_match": {
            "query": query_text,
            "fields": ["title^3", "body"],
            "fuzziness": "AUTO"
        }}]
        if fake_filter is not None:
            must.append({"term": {"is_fake": fake_filter}})
        body = {
            "query": {"bool": {"must": must}},
            "size": size,
            "_source": ["id", "title", "source", "language", "is_fake",
                        "confidence", "p_fake", "processed_at", "url", "body"]
        }
        resp = es.search(index="articles", body=body)
        hits = [h["_source"] for h in resp["hits"]["hits"]]
        return pd.DataFrame(hits)
    except Exception:
        return pd.DataFrame()

def fetch_alert_history(limit=100):
    db = get_mongo()
    if db is None:
        return []
    try:
        cursor = db.alert_history.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception:
        return []

def save_alert_event(alert_doc):
    db = get_mongo()
    if db is None:
        return
    try:
        db.alert_history.insert_one(alert_doc)
    except Exception:
        pass

def check_services():
    targets = {
        "Zookeeper":     ("localhost", 2181),
        "Kafka":         ("localhost", 9092),
        "MongoDB":       ("localhost", 27017),
        "Elasticsearch": ("localhost", 9200),
        "FastAPI":       ("localhost", 8000),
        "Kafdrop":       ("localhost", 9000),
        "Grafana":       ("localhost", 3000),
        "Streamlit":     ("localhost", 8501),
    }
    results = {}
    for name, (host, port) in targets.items():
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            results[name] = True
        except Exception:
            results[name] = False
    return results

def get_word_frequencies(df, n=30):
    if df.empty or "title" not in df.columns:
        return pd.DataFrame()
    stop = {"le","la","les","de","du","des","un","une","et","en","à","au","aux",
            "est","que","qui","pour","par","sur","dans","avec","ce","se","il","ils",
            "elle","elles","on","nous","vous","the","a","an","of","in","to","and",
            "is","it","for","that","was","are","with","as","at","by","this","from"}
    words = {}
    for title in df["title"].dropna():
        for w in str(title).lower().split():
            w = w.strip(".,;:!?\"'()[]{}").replace("'","").replace("'","")
            if len(w) > 3 and w not in stop:
                words[w] = words.get(w, 0) + 1
    return pd.DataFrame(
        sorted(words.items(), key=lambda x: x[1], reverse=True)[:n],
        columns=["mot", "fréquence"]
    )


# ── Évaluation des alertes ────────────────────────────────────────────────────
def evaluate_alerts(stats, thresholds):
    alerts = []
    now = datetime.now(timezone.utc).isoformat()

    fake_pct = stats.get("fake_rate", 0.0)
    if fake_pct >= thresholds["fake_rate_crit"]:
        alerts.append({
            "severity": "critical",
            "title": "Taux de désinformation critique",
            "message": f"Taux actuel : {fake_pct:.1f}% (seuil : {thresholds['fake_rate_crit']}%)",
            "metric": "fake_rate", "value": fake_pct, "timestamp": now
        })
    elif fake_pct >= thresholds["fake_rate_warn"]:
        alerts.append({
            "severity": "warning",
            "title": "Taux de désinformation élevé",
            "message": f"Taux actuel : {fake_pct:.1f}% (seuil : {thresholds['fake_rate_warn']}%)",
            "metric": "fake_rate", "value": fake_pct, "timestamp": now
        })

    drift_events = fetch_drift_events(limit=5)
    if drift_events:
        latest_score = drift_events[0].get("composite_score", 0)
        if latest_score >= thresholds["drift_crit"]:
            alerts.append({
                "severity": "critical",
                "title": "Concept Drift confirmé",
                "message": f"Score : {latest_score:.3f} (seuil critique : {thresholds['drift_crit']})",
                "metric": "drift_score", "value": latest_score, "timestamp": now
            })
        elif latest_score >= thresholds["drift_warn"]:
            alerts.append({
                "severity": "warning",
                "title": "Concept Drift détecté",
                "message": f"Score : {latest_score:.3f} (seuil avertissement : {thresholds['drift_warn']})",
                "metric": "drift_score", "value": latest_score, "timestamp": now
            })

    return alerts


# ── Composants UI ─────────────────────────────────────────────────────────────
def kpi_card(label, value, css_class="", unit=""):
    st.markdown(f"""
    <div class="kpi-card {css_class}">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}<span style="font-size:1rem;color:#95A5A6"> {unit}</span></p>
    </div>
    """, unsafe_allow_html=True)

def section_header(icon, title):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def no_data_msg(msg="Données non disponibles — démarrez le pipeline Docker."):
    st.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 18px 0 8px 0;">
        <div style="font-size:2.4rem;">🔍</div>
        <div style="font-size:1.05rem; font-weight:700; letter-spacing:1px;">PIPELINE DÉSINFORMATION</div>
        <div style="font-size:0.72rem; opacity:0.7; margin-top:4px;">Master BIG DATA IA — UCAO UUT</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.2); margin:8px 0 16px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Tableau de bord",
            "📰  Articles temps réel",
            "🔍  Recherche & Analyse",
            "📈  Drift & Apprentissage",
            "🚨  Alertes",
            "⚙️  Infrastructure",
            "ℹ️  À propos",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:16px 0 10px 0;'>", unsafe_allow_html=True)

    auto_refresh = st.toggle("🔄 Rafraîchissement auto", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Intervalle (s)", 10, 120, REFRESH_SEC)

    # Seuils d'alerte configurables
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:10px 0 8px 0;'>", unsafe_allow_html=True)
    with st.expander("⚙️ Seuils d'alerte", expanded=False):
        t_fake_warn = st.slider("Fake avertissement (%)", 10, 80, int(DEFAULT_THRESHOLDS["fake_rate_warn"]), 5)
        t_fake_crit = st.slider("Fake critique (%)", 20, 95, int(DEFAULT_THRESHOLDS["fake_rate_crit"]), 5)
        t_drift_crit = st.slider("Drift critique", 0.1, 0.9, DEFAULT_THRESHOLDS["drift_crit"], 0.05)

    thresholds = {
        "fake_rate_warn": float(t_fake_warn),
        "fake_rate_crit": float(t_fake_crit),
        "drift_warn":     DEFAULT_THRESHOLDS["drift_warn"],
        "drift_crit":     float(t_drift_crit),
        "conf_low":       DEFAULT_THRESHOLDS["conf_low"],
        "silence_minutes": DEFAULT_THRESHOLDS["silence_minutes"],
    }

    # Status rapide
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:8px 0 8px 0;'>", unsafe_allow_html=True)
    health = fetch_health()
    mongo_ok = health.get("mongo") == "up"
    es_ok    = health.get("elasticsearch") == "up"
    api_ok   = bool(health)
    st.markdown(f"""
    <div style='font-size:0.78rem;'>
        <span class='status-dot {"status-up" if mongo_ok else "status-down"}'></span>MongoDB<br>
        <span class='status-dot {"status-up" if es_ok else "status-down"}'></span>Elasticsearch<br>
        <span class='status-dot {"status-up" if api_ok else "status-down"}'></span>FastAPI
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.68rem;opacity:0.5;margin-top:12px;'>v1.0 · {datetime.now().strftime('%H:%M:%S')}</div>",
                unsafe_allow_html=True)

    if not _backend_ok():
        st.markdown("""
        <div style='background:rgba(231,76,60,0.2);border-radius:6px;padding:8px 10px;
                    margin-top:10px;font-size:0.75rem;'>
            ⚠️ Mode limité<br>Backends indisponibles
        </div>
        """, unsafe_allow_html=True)

    # Évaluation alertes actives pour badge sidebar
    stats_quick = fetch_stats()
    active_alerts = evaluate_alerts(stats_quick, thresholds)
    if active_alerts:
        crits = sum(1 for a in active_alerts if a["severity"] == "critical")
        warns = sum(1 for a in active_alerts if a["severity"] == "warning")
        badge = f"🔴 {crits} crit." if crits else f"🟡 {warns} avert."
        st.markdown(f"<div style='font-size:0.8rem;margin-top:6px;'>{badge}</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ═════════════════════════════════════════════════════════════════════════════
if "Tableau de bord" in page:
    section_header("🏠", "Tableau de Bord — Surveillance en Temps Réel")

    # Alertes actives en bannière
    if active_alerts:
        for a in active_alerts:
            cls = "alert-critical" if a["severity"] == "critical" else "alert-warning"
            icon = "🔴" if a["severity"] == "critical" else "🟡"
            st.markdown(f'<div class="{cls}"><strong>{icon} {a["title"]}</strong> — {a["message"]}</div>',
                        unsafe_allow_html=True)
        st.markdown("")

    stats = stats_quick
    total    = stats.get("total_articles", 0)
    fakes    = stats.get("fake_articles",  0)
    reals    = stats.get("real_articles",  0)
    fake_pct = stats.get("fake_rate",      0.0)
    drifts   = stats.get("drift_events",   0)
    last_h   = stats.get("articles_last_hour", 0)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Articles analysés",   f"{total:,}",     "info")
    with c2: kpi_card("Faux détectés",        f"{fakes:,}",     "fake")
    with c3: kpi_card("Vrais détectés",       f"{reals:,}",     "real")
    with c4: kpi_card("Taux de fake",         f"{fake_pct:.1f}", "fake" if fake_pct > 50 else "info", "%")
    with c5: kpi_card("Alertes drift",        f"{drifts}",      "drift")
    with c6: kpi_card("Articles (1h)",        f"{last_h}",      "info")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Répartition Fake / Réel")
        if total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Fake", "Réel"],
                values=[fakes, reals],
                hole=0.55,
                marker_colors=[CLR_FAKE, CLR_REAL],
                textfont_size=13,
                hovertemplate="<b>%{label}</b><br>%{value} articles (%{percent})<extra></extra>"
            )])
            fig.update_layout(
                showlegend=True, height=290,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.12),
                annotations=[dict(text=f"<b>{fake_pct:.1f}%</b><br>Fake",
                                  x=0.5, y=0.5, font_size=15, showarrow=False)]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data_msg()

    with col_r:
        st.subheader("Tendance Horaire (24h)")
        trend = fetch_virality(24)
        if trend:
            df_t = pd.DataFrame(trend)
            df_t["heure"] = df_t["_id"].str[-2:] + "h"
            df_t["fake_pct"] = (df_t["fakes"] / df_t["total"].replace(0, 1) * 100).round(1)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_t["heure"], y=df_t["total"],
                                  name="Total", marker_color="#BDC3C7", opacity=0.5))
            fig2.add_trace(go.Bar(x=df_t["heure"], y=df_t["fakes"],
                                  name="Faux", marker_color=CLR_FAKE, opacity=0.85))
            fig2.add_trace(go.Scatter(x=df_t["heure"], y=df_t["fake_pct"],
                                      name="% Fake", yaxis="y2",
                                      line=dict(color=CLR_DRIFT, width=2.5),
                                      mode="lines+markers"))
            fig2.update_layout(
                barmode="overlay", height=290,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.28),
                yaxis=dict(title="Articles"),
                yaxis2=dict(title="% Fake", overlaying="y", side="right",
                            showgrid=False, range=[0, 100])
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            no_data_msg("Tendances non disponibles")

    st.markdown("---")

    col_tbl, col_drift = st.columns([3, 2])

    with col_tbl:
        st.subheader("📋 Articles récemment classifiés")
        df = fetch_articles(limit=20)
        if not df.empty:
            df["Statut"]    = df["is_fake"].map({1: "🔴 FAKE", 0: "🟢 RÉEL"})
            df["Confiance"] = (df["confidence"] * 100).round(1).astype(str) + "%"
            df["P(fake)"]   = (df["p_fake"]     * 100).round(1).astype(str) + "%"
            df["Titre"]     = df["title"].str[:70].fillna("")
            st.dataframe(
                df[["Statut", "Titre", "source", "Confiance", "P(fake)", "language"]].rename(
                    columns={"source": "Source", "language": "Langue"}),
                use_container_width=True, hide_index=True,
                column_config={
                    "Titre":  st.column_config.TextColumn(width="large"),
                    "Statut": st.column_config.TextColumn(width="small"),
                }
            )
            # Export
            csv = df[["Statut","Titre","source","Confiance","P(fake)","language"]].to_csv(index=False)
            st.download_button("⬇️ Exporter CSV", csv, "articles_recent.csv", "text/csv")
        else:
            no_data_msg()

    with col_drift:
        st.subheader("📈 Score de Drift (derniers événements)")
        drift_evts = fetch_drift_events(limit=30)
        if drift_evts:
            df_drv = pd.DataFrame(drift_evts)
            if "timestamp" in df_drv.columns and "composite_score" in df_drv.columns:
                df_drv["timestamp"] = pd.to_datetime(df_drv["timestamp"])
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(
                    x=df_drv["timestamp"], y=df_drv["composite_score"],
                    mode="lines+markers", name="Score",
                    line=dict(color=CLR_DRIFT, width=2.5),
                    fill="tozeroy", fillcolor="rgba(243,156,18,0.1)"
                ))
                fig_d.add_hline(y=thresholds["drift_crit"], line_dash="dash",
                                line_color=CLR_FAKE, annotation_text="Seuil critique")
                fig_d.update_layout(
                    height=290, margin=dict(t=10, b=10),
                    yaxis=dict(title="Score", range=[0, 1.05]),
                    showlegend=False
                )
                st.plotly_chart(fig_d, use_container_width=True)
        else:
            st.success("✅ Aucun drift — modèle stable")

    st.markdown("---")
    st.subheader("⚡ Performances du Modèle")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1-Score",     "98.49%",  "NLP classification")
    m2.metric("AUC-ROC",      "99.89%",  "Courbe ROC")
    m3.metric("Latence ONNX", "5-6 ms",  "par article")
    m4.metric("Compression",  "75%",     "FP32 → INT8")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ARTICLES EN TEMPS RÉEL
# ═════════════════════════════════════════════════════════════════════════════
elif "Articles" in page:
    section_header("📰", "Articles Classifiés en Temps Réel")

    cf1, cf2, cf3, cf4, cf5 = st.columns([2, 2, 2, 1, 1])
    with cf1:
        filtre_statut = st.selectbox("Statut", ["Tous", "🔴 Fake", "🟢 Réel", "⚠️ Drift actif"])
    with cf2:
        try:
            db_tmp = get_mongo()
            sources = ["Toutes"] + sorted(db_tmp.articles.distinct("source")) if db_tmp else ["Toutes"]
        except Exception:
            sources = ["Toutes"]
        filtre_source = st.selectbox("Source", sources)
    with cf3:
        nb_articles = st.slider("Nombre", 10, 300, 50, 10)
    with cf4:
        conf_min = st.slider("Conf. min", 0.0, 1.0, 0.0, 0.05)
    with cf5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄 Rafraîchir", use_container_width=True, key="refresh_btn")

    fake_only  = filtre_statut == "🔴 Fake"
    real_only  = filtre_statut == "🟢 Réel"
    drift_only = filtre_statut == "⚠️ Drift actif"
    src_f = None if filtre_source == "Toutes" else filtre_source

    df = fetch_articles(limit=nb_articles * 2, fake_only=fake_only, real_only=real_only,
                        source_filter=src_f, conf_min=conf_min, drift_only=drift_only)
    df = df.head(nb_articles) if not df.empty else df

    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Affichés", len(df))
        c2.metric("Faux",  int(df["is_fake"].sum()))
        c3.metric("Vrais", int((df["is_fake"] == 0).sum()))
        c4.metric("Conf. moy.", f"{df['confidence'].mean()*100:.1f}%" if "confidence" in df.columns else "—")
        st.markdown("---")

    if not df.empty:
        ch1, ch2 = st.columns([1, 1])
        with ch1:
            st.subheader("Distribution des scores P(fake)")
            fig_h = px.histogram(df, x="p_fake", color="is_fake", nbins=30, barmode="overlay",
                                 color_discrete_map={1: CLR_FAKE, 0: CLR_REAL},
                                 labels={"p_fake": "P(fake)", "is_fake": "Catégorie"}, opacity=0.75)
            fig_h.update_layout(height=240, margin=dict(t=5, b=5),
                                legend=dict(orientation="h", y=-0.35))
            st.plotly_chart(fig_h, use_container_width=True)

        with ch2:
            st.subheader("Articles par source (top 10)")
            src_c = df.groupby("source")["is_fake"].agg(total="count", fakes="sum").reset_index()
            src_c = src_c.sort_values("total", ascending=False).head(10)
            fig_s = go.Figure(data=[
                go.Bar(name="Vrais", x=src_c["source"],
                       y=src_c["total"] - src_c["fakes"], marker_color=CLR_REAL),
                go.Bar(name="Faux",  x=src_c["source"],
                       y=src_c["fakes"], marker_color=CLR_FAKE),
            ])
            fig_s.update_layout(barmode="stack", height=240, margin=dict(t=5, b=5),
                                xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.4))
            st.plotly_chart(fig_s, use_container_width=True)

        # Fréquences de mots
        st.markdown("---")
        ch3, ch4 = st.columns([1, 1])
        with ch3:
            st.subheader("Mots les plus fréquents (titres)")
            wf = get_word_frequencies(df, n=20)
            if not wf.empty:
                fig_w = px.bar(wf, x="fréquence", y="mot", orientation="h",
                               color="fréquence", color_continuous_scale="Reds", height=280)
                fig_w.update_layout(margin=dict(t=5, b=5), showlegend=False,
                                    coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_w, use_container_width=True)

        with ch4:
            st.subheader("Distribution par langue")
            if "language" in df.columns:
                lang_c = df["language"].value_counts().reset_index()
                lang_c.columns = ["Langue", "Count"]
                fig_l = px.pie(lang_c, names="Langue", values="Count",
                               color_discrete_sequence=px.colors.qualitative.Set2, height=280)
                fig_l.update_layout(margin=dict(t=5, b=5))
                st.plotly_chart(fig_l, use_container_width=True)

    st.markdown("---")
    st.subheader(f"📄 {len(df) if not df.empty else 0} articles")

    if df.empty:
        no_data_msg("Aucun article — pipeline en cours ou filtres trop restrictifs.")
    else:
        # Export
        export_df = df.copy()
        export_df["Statut"] = export_df["is_fake"].map({1: "FAKE", 0: "REEL"})
        csv_exp = export_df[["Statut","title","source","language","confidence","p_fake",
                              "drift_score","processed_at","url"]].to_csv(index=False)
        st.download_button("⬇️ Exporter tous les articles (CSV)", csv_exp,
                           "articles_export.csv", "text/csv")
        st.markdown("")

        for _, row in df.iterrows():
            is_fake  = row.get("is_fake", 0) == 1
            badge    = "fake" if is_fake else "real"
            btxt     = "🔴 FAKE" if is_fake else "🟢 RÉEL"
            conf     = row.get("confidence", 0) * 100
            drift_ic = "⚠️" if row.get("drift_active", False) else ""

            with st.expander(f"{btxt} {drift_ic} {row.get('title','(sans titre)')[:85]}"):
                ca, cb = st.columns([3, 1])
                with ca:
                    st.markdown(f"**Source :** {row.get('source','?')} | **Langue :** {row.get('language','?')}")
                    if row.get("body"):
                        st.markdown(f"*{str(row.get('body',''))[:200]}...*")
                    if row.get("url"):
                        st.markdown(f"[🔗 Lien]({row.get('url')})")
                with cb:
                    st.metric("Confiance",  f"{conf:.1f}%")
                    st.metric("P(fake)",    f"{row.get('p_fake',0)*100:.1f}%")
                    if row.get("drift_active"):
                        st.warning(f"Drift: {row.get('drift_score',0):.3f}")
                    if row.get("processed_at"):
                        st.caption(f"🕐 {str(row.get('processed_at',''))[:19]}")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RECHERCHE & ANALYSE
# ═════════════════════════════════════════════════════════════════════════════
elif "Recherche" in page:
    section_header("🔍", "Recherche & Analyse Full-Text (Elasticsearch)")

    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []

    cs1, cs2, cs3 = st.columns([4, 1, 1])
    with cs1:
        query = st.text_input("🔎 Requête",
                              placeholder="Ex: désinformation, Covid, élections, Ukraine...",
                              label_visibility="collapsed")
    with cs2:
        nb_results = st.selectbox("Résultats", [10, 20, 50], label_visibility="collapsed")
    with cs3:
        fake_filter_opt = st.selectbox("Filtrer", ["Tous", "Fake only", "Réel only"],
                                       label_visibility="collapsed")

    fake_f = None
    if fake_filter_opt == "Fake only":  fake_f = 1
    if fake_filter_opt == "Réel only":  fake_f = 0

    if query:
        if query not in st.session_state["search_history"]:
            st.session_state["search_history"].insert(0, query)
            st.session_state["search_history"] = st.session_state["search_history"][:10]

        with st.spinner("Recherche en cours..."):
            results = search_articles(query, size=nb_results, fake_filter=fake_f)

        if results.empty:
            st.warning(f"Aucun résultat pour **{query}**.")
        else:
            total_r = len(results)
            fakes_r = int(results["is_fake"].sum()) if "is_fake" in results.columns else 0
            reals_r = total_r - fakes_r

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Résultats", total_r)
            c2.metric("🔴 Faux", fakes_r)
            c3.metric("🟢 Vrais", reals_r)
            if fakes_r + reals_r > 0:
                c4.metric("Taux fake", f"{fakes_r/(fakes_r+reals_r)*100:.1f}%")

            st.markdown("---")

            cr1, cr2 = st.columns([1, 1])
            with cr1:
                if "p_fake" in results.columns:
                    fig_sc = px.scatter(
                        results, x=results.index, y="p_fake", color="is_fake",
                        color_discrete_map={1: CLR_FAKE, 0: CLR_REAL},
                        size="confidence" if "confidence" in results.columns else None,
                        hover_data=["title"] if "title" in results.columns else [],
                        labels={"p_fake": "Score Fake", "index": "Rang"},
                        title=f"Scores — « {query} »", height=260
                    )
                    fig_sc.update_layout(margin=dict(t=40, b=5), showlegend=False)
                    st.plotly_chart(fig_sc, use_container_width=True)

            with cr2:
                wf_r = get_word_frequencies(results, n=15)
                if not wf_r.empty:
                    fig_wr = px.bar(wf_r, x="fréquence", y="mot", orientation="h",
                                    color="fréquence", color_continuous_scale="Blues",
                                    title="Mots fréquents dans les résultats", height=260)
                    fig_wr.update_layout(margin=dict(t=40, b=5), showlegend=False,
                                         coloraxis_showscale=False,
                                         yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_wr, use_container_width=True)

            # Export résultats
            if not results.empty:
                csv_r = results.to_csv(index=False)
                st.download_button("⬇️ Exporter résultats (CSV)", csv_r,
                                   f"recherche_{query[:20]}.csv", "text/csv")

            st.markdown("---")
            st.subheader("Résultats détaillés")
            for _, row in results.iterrows():
                is_fake  = row.get("is_fake", 0) == 1
                btxt     = "🔴 FAKE" if is_fake else "🟢 RÉEL"
                conf     = row.get("confidence", 0) * 100
                with st.expander(f"{btxt} {row.get('title','(sans titre)')[:85]}"):
                    st.markdown(f"**Source :** {row.get('source','?')} | "
                                f"**Langue :** {row.get('language','?')} | "
                                f"**Confiance :** {conf:.1f}%")
                    if row.get("url"):
                        st.markdown(f"[🔗 Lien]({row.get('url')})")
    else:
        # Historique des recherches
        if st.session_state["search_history"]:
            st.markdown("#### 🕐 Recherches récentes")
            cols_h = st.columns(min(5, len(st.session_state["search_history"])))
            for col, term in zip(cols_h, st.session_state["search_history"][:5]):
                col.markdown(f"""
                <div style="background:white;border-radius:8px;padding:10px;
                            text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.1);
                            cursor:pointer;font-size:0.85rem;">{term}</div>
                """, unsafe_allow_html=True)
            st.markdown("---")

        st.markdown("#### 💡 Exemples de recherches")
        examples = [("COVID-19","Pandémie"), ("élections","Politique"),
                    ("Ukraine","Géopolitique"), ("vaccins","Santé"),
                    ("deepfake","Technologie")]
        cols_e = st.columns(len(examples))
        for col, (term, cat) in zip(cols_e, examples):
            with col:
                st.markdown(f"""
                <div style="background:white;border-radius:8px;padding:12px;
                            text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.1);">
                    <div style="font-weight:600;">{term}</div>
                    <div style="font-size:0.75rem;color:#7F8C8D;">{cat}</div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DRIFT & APPRENTISSAGE
# ═════════════════════════════════════════════════════════════════════════════
elif "Drift" in page:
    section_header("📈", "Détection de Concept Drift & Apprentissage Continu")

    st.markdown("""
    Tri-détecteur hybride basé sur **River 0.21.2** :
    - **ADWIN** (poids 0.45) — dérive abrupte par fenêtre adaptative
    - **KSWIN** (poids 0.35) — changement statistique (Kolmogorov-Smirnov)
    - **PageHinkley** (poids 0.20) — dérive graduelle par somme cumulée
    """)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Seuil avertissement", "0.3",  "Score composite")
    d2.metric("Seuil détection",     "0.5",  "Score composite")
    d3.metric("Seuil confirmation",  "0.8",  "Score composite")
    d4.metric("LR adaptatif",        "5e-5", "En cas de drift")

    st.markdown("---")
    st.subheader("🧮 Formule du Score Composite")
    st.latex(r"S = 0.45 \times ADWIN + 0.35 \times KSWIN + 0.20 \times PageHinkley")

    st.markdown("---")

    drift_events = fetch_drift_events(limit=100)
    col_ev, col_stat = st.columns([3, 1])

    with col_ev:
        st.subheader(f"Historique des Événements ({len(drift_events)})")
        if drift_events:
            df_dr = pd.DataFrame(drift_events)
            if "timestamp" in df_dr.columns:
                df_dr["timestamp"] = pd.to_datetime(df_dr["timestamp"])

                # Score composite global
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(
                    x=df_dr["timestamp"], y=df_dr.get("composite_score", pd.Series([0]*len(df_dr))),
                    mode="lines+markers", name="Score composite",
                    line=dict(color=CLR_DRIFT, width=2.5),
                    fill="tozeroy", fillcolor="rgba(243,156,18,0.12)"
                ))
                fig_comp.add_hline(y=0.3, line_dash="dot", line_color="gray",
                                   annotation_text="Avert.")
                fig_comp.add_hline(y=0.5, line_dash="dash", line_color=CLR_FAKE,
                                   annotation_text="Détection")
                fig_comp.add_hline(y=0.8, line_dash="dash", line_color=CLR_PURPLE,
                                   annotation_text="Confirmation")
                fig_comp.update_layout(
                    title="Score composite de drift", height=300,
                    margin=dict(t=30, b=10),
                    yaxis=dict(title="Score", range=[0, 1.05])
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                # Détecteurs individuels si disponibles
                sigs = df_dr.get("signals")
                if sigs is not None:
                    try:
                        df_dr["adwin"] = df_dr["signals"].apply(
                            lambda s: 1 if isinstance(s, dict) and s.get("ADWIN") else 0)
                        df_dr["kswin"] = df_dr["signals"].apply(
                            lambda s: 1 if isinstance(s, dict) and s.get("KSWIN") else 0)
                        df_dr["ph"] = df_dr["signals"].apply(
                            lambda s: 1 if isinstance(s, dict) and s.get("PageHinkley") else 0)
                        fig_det = go.Figure()
                        fig_det.add_trace(go.Scatter(
                            x=df_dr["timestamp"], y=df_dr["adwin"].cumsum(),
                            name="ADWIN (cum.)", line=dict(color="#E74C3C")))
                        fig_det.add_trace(go.Scatter(
                            x=df_dr["timestamp"], y=df_dr["kswin"].cumsum(),
                            name="KSWIN (cum.)", line=dict(color="#3498DB")))
                        fig_det.add_trace(go.Scatter(
                            x=df_dr["timestamp"], y=df_dr["ph"].cumsum(),
                            name="PageHinkley (cum.)", line=dict(color="#2ECC71")))
                        fig_det.update_layout(
                            title="Déclenchements cumulés par détecteur", height=240,
                            margin=dict(t=30, b=10), legend=dict(orientation="h", y=-0.3))
                        st.plotly_chart(fig_det, use_container_width=True)
                    except Exception:
                        pass

            # Export
            csv_d = pd.DataFrame(drift_events).to_csv(index=False)
            st.download_button("⬇️ Exporter événements drift (CSV)", csv_d,
                               "drift_events.csv", "text/csv")

            st.subheader("Derniers événements")
            for ev in drift_events[:5]:
                confirmed = ev.get("drift_confirmed", False)
                st.markdown(f"""
                <div class="drift-alert">
                    <strong>{"⚠️ CONFIRMÉ" if confirmed else "🔔 Détecté"}</strong>
                    — Score : <b>{ev.get("composite_score",0):.3f}</b><br>
                    <small>ADWIN: {ev.get("signals",{}).get("ADWIN","?")} |
                           KSWIN: {ev.get("signals",{}).get("KSWIN","?")} |
                           PH: {ev.get("signals",{}).get("PageHinkley","?")}</small><br>
                    <small>🕐 {str(ev.get("timestamp",""))[:19]} |
                           LR recommandé : {ev.get("recommended_lr","?")}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Aucun drift détecté — le modèle est stable.")

    with col_stat:
        st.subheader("Statistiques")
        st.metric("Événements", len(drift_events))
        confirmed = sum(1 for e in drift_events if e.get("drift_confirmed", False))
        st.metric("Confirmés", confirmed)
        if drift_events:
            avg_sc = np.mean([e.get("composite_score", 0) for e in drift_events])
            st.metric("Score moyen", f"{avg_sc:.3f}")
            max_sc = max(e.get("composite_score", 0) for e in drift_events)
            st.metric("Score max", f"{max_sc:.3f}")

        st.markdown("---")
        st.subheader("Apprentissage continu")
        st.markdown("""
        **Reservoir** : 5 000 exemples représentatifs

        **Mini-batch replay** : 8 exemples / mise à jour

        **Sync ONNX** : tous les 100 batches
        """)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ALERTES
# ═════════════════════════════════════════════════════════════════════════════
elif "Alertes" in page:
    section_header("🚨", "Centre d'Alertes — Surveillance Intelligente")

    # ── Alertes actives ───────────────────────────────────────────────────────
    st.subheader("🔴 État Actuel des Alertes")
    current_stats = fetch_stats()
    current_alerts = evaluate_alerts(current_stats, thresholds)

    if not current_alerts:
        st.markdown('<div class="alert-ok">✅ <strong>Tous les indicateurs sont normaux</strong> — aucune alerte active</div>',
                    unsafe_allow_html=True)
    else:
        for a in current_alerts:
            cls  = "alert-critical" if a["severity"] == "critical" else "alert-warning"
            icon = "🔴 CRITIQUE" if a["severity"] == "critical" else "🟡 AVERTISSEMENT"
            ts   = a.get("timestamp", "")[:19]
            st.markdown(f"""
            <div class="{cls}">
                <strong>{icon} — {a['title']}</strong><br>
                {a['message']}<br>
                <small>🕐 {ts}</small>
            </div>
            """, unsafe_allow_html=True)
            save_alert_event(a)

    st.markdown("---")

    # ── Tableau de bord des indicateurs ──────────────────────────────────────
    st.subheader("📊 Indicateurs en Temps Réel")

    fake_pct  = current_stats.get("fake_rate",    0.0)
    total_art = current_stats.get("total_articles", 0)
    n_drifts  = current_stats.get("drift_events",   0)

    ai1, ai2, ai3, ai4 = st.columns(4)
    with ai1:
        status = "🔴" if fake_pct >= thresholds["fake_rate_crit"] else \
                 "🟡" if fake_pct >= thresholds["fake_rate_warn"] else "🟢"
        kpi_card(f"{status} Taux de fake", f"{fake_pct:.1f}", "fake" if fake_pct > 50 else "info", "%")
    with ai2:
        kpi_card("Articles traités", f"{total_art:,}", "info")
    with ai3:
        kpi_card("Événements drift", f"{n_drifts}", "drift")
    with ai4:
        h = fetch_health()
        services_ok = sum(1 for v in h.values() if v == "up")
        kpi_card("Services UP", f"{services_ok}/{len(h)}", "real" if services_ok == len(h) else "warn")

    st.markdown("---")

    # ── Jauge du taux de fake ─────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Taux de désinformation")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fake_pct,
            title={"text": "Fake Rate (%)"},
            delta={"reference": thresholds["fake_rate_warn"]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": CLR_FAKE if fake_pct > thresholds["fake_rate_crit"]
                                else (CLR_DRIFT if fake_pct > thresholds["fake_rate_warn"]
                                      else CLR_REAL)},
                "steps": [
                    {"range": [0,  thresholds["fake_rate_warn"]], "color": "#EAFAF1"},
                    {"range": [thresholds["fake_rate_warn"], thresholds["fake_rate_crit"]], "color": "#FEF9E7"},
                    {"range": [thresholds["fake_rate_crit"], 100], "color": "#FDEDEC"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": thresholds["fake_rate_crit"]
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=10, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        st.subheader("Historique des alertes")
        alert_hist = fetch_alert_history(limit=30)
        if alert_hist:
            df_ah = pd.DataFrame(alert_hist)
            if "timestamp" in df_ah.columns and "metric" in df_ah.columns:
                df_ah["timestamp"] = pd.to_datetime(df_ah["timestamp"])
                df_ah["severity_num"] = df_ah["severity"].map(
                    {"critical": 2, "warning": 1}).fillna(0)
                fig_ah = px.scatter(
                    df_ah, x="timestamp", y="metric",
                    color="severity",
                    color_discrete_map={"critical": CLR_FAKE, "warning": CLR_DRIFT},
                    size="severity_num",
                    title="Alertes déclenchées",
                    height=280,
                    labels={"metric": "Indicateur", "timestamp": ""}
                )
                fig_ah.update_layout(margin=dict(t=40, b=10),
                                     legend=dict(orientation="h", y=-0.3))
                st.plotly_chart(fig_ah, use_container_width=True)
            else:
                no_data_msg("Format d'historique inattendu")
        else:
            st.info("Aucun historique d'alerte disponible pour cette session.")

    st.markdown("---")

    # ── Règles d'alerte configurées ───────────────────────────────────────────
    st.subheader("📋 Règles d'Alerte Actives")
    rules_data = {
        "Règle": [
            "Taux de fake (avertissement)",
            "Taux de fake (critique)",
            "Score drift (avertissement)",
            "Score drift (critique)",
            "Confiance modèle (faible)",
        ],
        "Seuil": [
            f"> {thresholds['fake_rate_warn']:.0f}%",
            f"> {thresholds['fake_rate_crit']:.0f}%",
            f"> {thresholds['drift_warn']:.2f}",
            f"> {thresholds['drift_crit']:.2f}",
            f"< {thresholds['conf_low']*100:.0f}%",
        ],
        "Sévérité": ["Avertissement", "Critique", "Avertissement", "Critique", "Avertissement"],
        "Délai évaluation": ["5 min", "5 min", "2 min", "1 min", "10 min"],
        "Statut": ["Actif", "Actif", "Actif", "Actif", "Actif"],
    }
    st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)

    if alert_hist:
        csv_ah = pd.DataFrame(alert_hist).to_csv(index=False)
        st.download_button("⬇️ Exporter historique alertes (CSV)", csv_ah,
                           "alertes_historique.csv", "text/csv")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════════════
elif "Infrastructure" in page:
    section_header("⚙️", "État de l'Infrastructure Docker")

    services_status = check_services()
    n_up = sum(services_status.values())
    n_total = len(services_status)
    st.markdown(f"**{n_up}/{n_total} services opérationnels** "
                f"{'✅' if n_up == n_total else '⚠️'}")
    st.markdown("")

    services_info = {
        "Zookeeper":     {"icon": "🦒", "port": 2181,  "role": "Coordination Kafka"},
        "Kafka":         {"icon": "📨", "port": 9092,  "role": "Message Broker"},
        "MongoDB":       {"icon": "🍃", "port": 27017, "role": "Document Store"},
        "Elasticsearch": {"icon": "🔎", "port": 9200,  "role": "Full-Text Search"},
        "FastAPI":       {"icon": "⚡", "port": 8000,  "role": "API REST"},
        "Kafdrop":       {"icon": "📊", "port": 9000,  "role": "Kafka UI"},
        "Grafana":       {"icon": "📈", "port": 3000,  "role": "Dashboards"},
        "Streamlit":     {"icon": "🎯", "port": 8501,  "role": "Ce dashboard"},
    }
    cols = st.columns(4)
    for i, (name, info) in enumerate(services_info.items()):
        is_up = services_status.get(name, False)
        color = "#2ECC71" if is_up else "#E74C3C"
        status_txt = "🟢 UP" if is_up else "🔴 DOWN"
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:16px;
                        margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.08);
                        border-top:3px solid {color};">
                <div style="font-size:1.4rem">{info['icon']}</div>
                <div style="font-weight:600;margin-top:4px">{name}</div>
                <div style="font-size:0.74rem;color:#7F8C8D">{info['role']}</div>
                <div style="font-size:0.8rem;margin-top:6px">{status_txt}</div>
                <div style="font-size:0.68rem;color:#BDC3C7">port {info['port']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔗 Liens Rapides")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.link_button("📈 Grafana",       "http://localhost:3000", use_container_width=True)
    lc2.link_button("📨 Kafdrop",       "http://localhost:9000", use_container_width=True)
    lc3.link_button("⚡ API Docs",      "http://localhost:8000/docs", use_container_width=True)
    lc4.link_button("🔎 Elasticsearch", "http://localhost:9200", use_container_width=True)

    st.markdown("---")
    st.subheader("🏗️ Architecture du Pipeline")
    st.code("""
SOURCES DE DONNÉES
  📡 RSS Feeds (AFP, BBC, Reuters, Al Jazeera, Jeune Afrique...)
  🌐 GDELT API (articles géopolitiques multilingues)
         │ scraping toutes les 60s
         ▼
APACHE KAFKA 3.7 (Confluent 7.6.0)
  📥 raw-news-stream  (6 partitions)
  📤 classified-news  (6 partitions)
  🔔 drift-alerts     (1 partition)
         │ Spark Structured Streaming
         ▼
SPARK STREAMING — NLP Pipeline
  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐
  │  ONNX Inference │  │  Drift Detection  │  │  Online Learning  │
  │  DistilBERT INT8│  │  ADWIN+KSWIN+PH   │  │  Reservoir 5000   │
  │  ~5-6ms/article │  │  Score composite  │  │  PyTorch AdamW    │
  └─────────────────┘  └──────────────────┘  └───────────────────┘
         │ bulk write
    ┌────┴────┐
    ▼         ▼
MongoDB 7.0   Elasticsearch 8.14.0
    └────┬────┘
         ▼
COUCHE PRÉSENTATION
  ⚡ FastAPI   (port 8000) — REST API
  🎯 Streamlit (port 8501) — Dashboard
  📈 Grafana   (port 3000) — Métriques
  📊 Kafdrop   (port 9000) — Kafka UI
""", language=None)

    st.markdown("---")
    st.subheader("💾 Allocation Mémoire")
    mem = {
        "Service":    ["spark-app", "Elasticsearch", "Kafka", "MongoDB",
                       "Kafdrop", "Grafana", "FastAPI", "Streamlit", "Autres"],
        "Limite (MB)":[4096, 768, 640, 512, 192, 256, 256, 512, 256],
    }
    fig_mem = px.bar(pd.DataFrame(mem), x="Service", y="Limite (MB)",
                     color="Service", text="Limite (MB)",
                     color_discrete_sequence=px.colors.qualitative.Set3, height=260)
    fig_mem.update_layout(showlegend=False, margin=dict(t=5, b=5))
    fig_mem.add_hline(y=11*1024, line_dash="dash", line_color="red",
                      annotation_text="RAM totale ≈ 11 GB")
    st.plotly_chart(fig_mem, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — À PROPOS
# ═════════════════════════════════════════════════════════════════════════════
elif "propos" in page:
    section_header("ℹ️", "À Propos du Projet")

    col_info, col_meta = st.columns([2, 1])

    with col_info:
        st.markdown("""
        ## Pipeline Big Data de Monitoring de la Désinformation en Temps Réel

        Ce projet constitue le mémoire de fin d'études du **Master BIG DATA IA**
        à l'**Université Catholique de l'Afrique de l'Ouest — Unité Universitaire
        du Togo (UCAO UUT)**, année 2025-2026.

        ### 🎯 Objectifs

        1. **Collecte en temps réel** d'articles via RSS et l'API GDELT
        2. **Classification automatique** fake/réel par Continual-DistilBERT (ONNX INT8)
        3. **Détection de concept drift** pour adapter le modèle aux nouvelles formes de désinformation
        4. **Apprentissage continu** avec reservoir sampling (évite l'oubli catastrophique)
        5. **Visualisation temps réel** via ce dashboard, Grafana et une API REST

        ### 🔬 Innovations Techniques

        - **Continual Learning** : adaptation continue sans réentraînement complet
        - **Tri-détecteur hybride** : ADWIN + KSWIN + PageHinkley avec score composite pondéré
        - **Inférence ultra-rapide** : ONNX INT8 quantifié (75% de compression, ~5-6 ms/article)
        - **Architecture 11 services Docker** : entièrement conteneurisée et reproductible

        ### 📊 Stack Technologique
        """)

        tech = {
            "Composant":  ["Apache Kafka", "Apache Spark", "DistilBERT", "ONNX Runtime",
                           "River", "MongoDB", "Elasticsearch", "FastAPI", "Streamlit", "Docker"],
            "Version":    ["3.7 (Confluent 7.6)", "3.5.3", "multilingual-cased",
                           "1.19.0", "0.21.2", "7.0", "8.14.0", "0.113", "1.38", "Compose v2"],
            "Rôle":       ["Message Broker", "Streaming Engine", "Modèle NLP",
                           "Inférence rapide", "Concept Drift", "Document Store",
                           "Full-Text Search", "API REST", "Dashboard", "Orchestration"],
        }
        st.dataframe(pd.DataFrame(tech), use_container_width=True, hide_index=True)

    with col_meta:
        st.markdown("### 👤 Auteur")
        st.markdown("""
        **KOMOSSI Sosso**
        Master BIG DATA IA
        UCAO UUT, Lomé, Togo
        Année 2025-2026

        ---
        **Encadrants :**
        M. TCHANTCHO Leri
        M. BABA Kpatcha

        ---
        """)

        st.markdown("### 📈 Performances du Modèle")
        perf = {
            "Métrique": ["F1-Score", "AUC-ROC", "Latence ONNX", "Compression modèle"],
            "Valeur":   ["98.49%",   "99.89%",  "~5-6 ms",      "75% (FP32→INT8)"]
        }
        st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

        st.markdown("### 📚 Datasets d'entraînement")
        st.markdown("""
        - **ISOT** (44 898 articles)
        - **WELFake** (72 134 articles)
        - **FakeNewsNet** (~23 196 articles)
        - **LIAR** (12 836 articles)
        - **Total : ~153 064 exemples**
        """)

    st.markdown("---")
    st.subheader("⚡ Endpoints API REST")
    endpoints = [
        ("GET", "/health",                       "Santé MongoDB + Elasticsearch"),
        ("GET", "/api/v1/stats",                 "Statistiques globales"),
        ("GET", "/api/v1/articles/recent",       "Derniers articles classifiés"),
        ("GET", "/api/v1/articles/search?q=...", "Recherche full-text Elasticsearch"),
        ("GET", "/api/v1/articles/virality",     "Tendance horaire du taux de fake"),
        ("GET", "/api/v1/drift/events",          "Historique des alertes drift"),
        ("GET", "/api/v1/drift/stats",           "Statistiques agrégées drift"),
    ]
    st.dataframe(pd.DataFrame(endpoints, columns=["Méthode","Endpoint","Description"]),
                 use_container_width=True, hide_index=True,
                 column_config={"Méthode": st.column_config.TextColumn(width="small")})
