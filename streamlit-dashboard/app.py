"""
Dashboard Streamlit — Pipeline Big Data de Monitoring de la Désinformation en Temps Réel
Auteur   : KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT 2025-2026
Encadrant: M. TCHANTCHO Leri & M. BABA Kpatcha
"""

import os, time, json, requests
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
from elasticsearch import Elasticsearch

# ── Configuration ─────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://mongodb:27017")
MONGO_DB    = os.getenv("MONGO_DB",    "disinformation_db")
ES_HOST     = os.getenv("ES_HOST",     "http://elasticsearch:9200")
API_BASE    = os.getenv("API_BASE",    "http://api:8000")
REFRESH_SEC = int(os.getenv("REFRESH_SEC", "30"))

# ── Palette de couleurs ───────────────────────────────────────────────────────
CLR_FAKE    = "#E74C3C"   # rouge — fake
CLR_REAL    = "#2ECC71"   # vert  — real
CLR_DRIFT   = "#F39C12"   # orange — drift
CLR_PRIMARY = "#2C3E50"   # bleu nuit — primaire
CLR_BG      = "#F8F9FA"   # fond clair

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Désinformation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Pipeline Big Data Désinformation — KOMOSSI Sosso, Master 2 IBDIA UCAO-UUT"
    }
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* En-tête global */
[data-testid="stAppViewContainer"] { background: #F8F9FA; }

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 5px solid #2C3E50;
    margin-bottom: 12px;
}
.kpi-card.fake { border-left-color: #E74C3C; }
.kpi-card.real { border-left-color: #2ECC71; }
.kpi-card.drift{ border-left-color: #F39C12; }
.kpi-card.info { border-left-color: #3498DB; }
.kpi-value  { font-size: 2.2rem; font-weight: 700; color: #2C3E50; margin: 0; }
.kpi-label  { font-size: 0.85rem; color: #7F8C8D; font-weight: 500; margin: 0; letter-spacing: 0.5px; }
.kpi-delta  { font-size: 0.8rem; margin-top: 4px; }

/* Badge article */
.badge-fake { background:#E74C3C; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }
.badge-real { background:#2ECC71; color:white; border-radius:4px; padding:2px 8px; font-size:0.75rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a252f 0%, #2C3E50 100%);
}
[data-testid="stSidebar"] * { color: #ECF0F1 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 2px 0;
    display: block;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.1); }

/* Section header */
.section-header {
    background: linear-gradient(135deg, #2C3E50, #3498DB);
    color: white;
    padding: 16px 24px;
    border-radius: 10px;
    margin-bottom: 20px;
    font-size: 1.3rem;
    font-weight: 600;
}

/* Article card */
.article-card {
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #BDC3C7;
}
.article-card.fake { border-left-color: #E74C3C; }
.article-card.real { border-left-color: #2ECC71; }
.article-title { font-weight: 600; font-size: 0.95rem; color: #2C3E50; }
.article-meta  { font-size: 0.78rem; color: #95A5A6; margin-top: 4px; }

/* Alerte drift */
.drift-alert {
    background: #FEF9E7;
    border: 1px solid #F39C12;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* Status indicator */
.status-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 6px;
}
.status-up   { background: #2ECC71; }
.status-down { background: #E74C3C; }
</style>
""", unsafe_allow_html=True)


# ── Connexions DB (mis en cache) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_mongo():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return client[MONGO_DB]

@st.cache_resource(show_spinner=False)
def get_es():
    return Elasticsearch(ES_HOST, request_timeout=10)


# ── Fonctions de données ──────────────────────────────────────────────────────
def fetch_stats():
    try:
        r = requests.get(f"{API_BASE}/api/v1/stats", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def fetch_articles(limit=100, fake_only=False, source_filter=None):
    try:
        db = get_mongo()
        query = {}
        if fake_only:   query["is_fake"] = 1
        if source_filter: query["source"] = source_filter
        cursor = db.articles.find(
            query,
            {"_id": 0, "id": 1, "title": 1, "source": 1, "language": 1,
             "is_fake": 1, "confidence": 1, "p_fake": 1, "drift_score": 1,
             "drift_active": 1, "processed_at": 1, "url": 1, "body": 1}
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
    try:
        db = get_mongo()
        cursor = db.drift_events.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception:
        return []

def search_articles(query_text, size=20):
    try:
        es = get_es()
        body = {
            "query": {"multi_match": {
                "query": query_text,
                "fields": ["title^3", "body"],
                "fuzziness": "AUTO"
            }},
            "size": size,
            "_source": ["id", "title", "source", "language", "is_fake",
                        "confidence", "p_fake", "processed_at", "url"]
        }
        resp = es.search(index="articles", body=body)
        hits = [h["_source"] for h in resp["hits"]["hits"]]
        return pd.DataFrame(hits)
    except Exception:
        return pd.DataFrame()

def check_docker_services():
    services = {
        "Zookeeper": ("localhost", 2181),
        "Kafka": ("localhost", 9092),
        "MongoDB": ("localhost", 27017),
        "Elasticsearch": ("localhost", 9200),
        "FastAPI": ("localhost", 8000),
        "Kafdrop": ("localhost", 9000),
        "Grafana": ("localhost", 3000),
    }
    import socket
    statuses = {}
    for name, (host, port) in services.items():
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            statuses[name] = True
        except Exception:
            statuses[name] = False
    return statuses


# ── Composants réutilisables ──────────────────────────────────────────────────
def kpi_card(label, value, css_class="", unit=""):
    st.markdown(f"""
    <div class="kpi-card {css_class}">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}<span style="font-size:1rem;color:#95A5A6"> {unit}</span></p>
    </div>
    """, unsafe_allow_html=True)

def section_header(icon, title):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Navigation
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.5rem;">🔍</div>
        <div style="font-size:1.1rem; font-weight:700; letter-spacing:1px;">PIPELINE DÉSINFORMATION</div>
        <div style="font-size:0.75rem; opacity:0.7; margin-top:4px;">Master 2 IBDIA — UCAO-UUT</div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0 20px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠  Tableau de bord",
            "📰  Articles temps réel",
            "🔍  Recherche & Analyse",
            "📈  Drift & Apprentissage",
            "⚙️  Infrastructure",
            "ℹ️  À propos du projet",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:20px 0 10px 0;'>", unsafe_allow_html=True)

    # Auto-refresh toggle
    auto_refresh = st.toggle("🔄 Rafraîchissement auto", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Intervalle (secondes)", 10, 120, REFRESH_SEC)
        st.markdown(f"<div style='font-size:0.75rem;opacity:0.7;'>Prochain refresh dans {refresh_interval}s</div>", unsafe_allow_html=True)

    # Health status mini
    st.markdown("<hr style='border-color:rgba(255,255,255,0.2); margin:10px 0 10px 0;'>", unsafe_allow_html=True)
    health = fetch_health()
    mongo_ok = health.get("mongo") == "up"
    es_ok    = health.get("elasticsearch") == "up"
    st.markdown(f"""
    <div style='font-size:0.8rem;'>
        <span class='status-dot {"status-up" if mongo_ok else "status-down"}'></span>MongoDB<br>
        <span class='status-dot {"status-up" if es_ok else "status-down"}'></span>Elasticsearch<br>
        <span class='status-dot status-up'></span>API FastAPI
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.7rem;opacity:0.5;margin-top:16px;'>v1.0 • {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════════════════
if "Tableau de bord" in page:
    section_header("🏠", "Tableau de Bord — Surveillance en Temps Réel")

    stats = fetch_stats()
    total    = stats.get("total_articles", 0)
    fakes    = stats.get("fake_articles", 0)
    reals    = stats.get("real_articles", 0)
    fake_pct = stats.get("fake_rate", 0.0)
    drifts   = stats.get("drift_events", 0)
    last_h   = stats.get("articles_last_hour", 0)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Articles analysés", f"{total:,}", "info")
    with c2: kpi_card("Faux détectés",     f"{fakes:,}", "fake")
    with c3: kpi_card("Vrais détectés",    f"{reals:,}", "real")
    with c4: kpi_card("Taux de fake",      f"{fake_pct:.1f}", "", "%")
    with c5: kpi_card("Alertes drift",     f"{drifts}",  "drift")

    st.markdown("---")

    # ── Graphiques principaux ─────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Répartition Fake / Real")
        if total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["Fake 🔴", "Real 🟢"],
                values=[fakes, reals],
                hole=0.55,
                marker_colors=[CLR_FAKE, CLR_REAL],
                textfont_size=14,
                hovertemplate="<b>%{label}</b><br>%{value} articles<br>%{percent}<extra></extra>"
            )])
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.1),
                annotations=[dict(
                    text=f"<b>{fake_pct:.1f}%</b><br>Fake",
                    x=0.5, y=0.5, font_size=16, showarrow=False
                )]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée disponible")

    with col_right:
        st.subheader("Tendance Horaire des Faux (24h)")
        trend = fetch_virality(24)
        if trend:
            df_t = pd.DataFrame(trend)
            df_t["heure"] = df_t["_id"].str[-2:] + "h"
            df_t["fake_rate"] = (df_t["fakes"] / df_t["total"] * 100).round(1)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_t["heure"], y=df_t["total"],
                name="Total", marker_color="#BDC3C7", opacity=0.6
            ))
            fig2.add_trace(go.Bar(
                x=df_t["heure"], y=df_t["fakes"],
                name="Faux", marker_color=CLR_FAKE, opacity=0.85
            ))
            fig2.add_trace(go.Scatter(
                x=df_t["heure"], y=df_t["fake_rate"],
                name="% Fake", yaxis="y2",
                line=dict(color=CLR_DRIFT, width=2.5),
                mode="lines+markers"
            ))
            fig2.update_layout(
                barmode="overlay",
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.25),
                yaxis=dict(title="Articles"),
                yaxis2=dict(title="% Fake", overlaying="y", side="right",
                            showgrid=False, range=[0, 100]),
                xaxis=dict(title="Heure")
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Données de tendance non disponibles")

    st.markdown("---")

    # ── Derniers articles (tableau) ───────────────────────────────────────────
    st.subheader("📋 Articles récemment classifiés")
    df = fetch_articles(limit=20)
    if not df.empty:
        df["Statut"]     = df["is_fake"].map({1: "🔴 FAKE", 0: "🟢 RÉEL"})
        df["Confiance"]  = (df["confidence"] * 100).round(1).astype(str) + "%"
        df["Score Fake"] = (df["p_fake"] * 100).round(1).astype(str) + "%"
        df["Titre"]      = df["title"].str[:80] + "..."
        df["Source"]     = df["source"]
        display_cols = ["Statut", "Titre", "Source", "Confiance", "Score Fake", "language"]
        st.dataframe(
            df[display_cols].rename(columns={"language": "Langue"}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Statut": st.column_config.TextColumn(width="small"),
                "Titre":  st.column_config.TextColumn(width="large"),
            }
        )
    else:
        st.info("En attente des premières données...")

    # ── Métriques de performance ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Métriques de Performance du Modèle")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F1-Score", "98.49%", "NLP classification")
    m2.metric("AUC-ROC", "99.89%", "Courbe ROC")
    m3.metric("Latence ONNX", "5-6 ms", "par article")
    m4.metric("Compression", "75%", "FP32 → INT8")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ARTICLES EN TEMPS RÉEL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Articles" in page:
    section_header("📰", "Articles Classifiés en Temps Réel")

    # ── Filtres ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3, col_f4 = st.columns([2, 2, 2, 1])

    with col_f1:
        filtre_statut = st.selectbox(
            "Filtrer par statut",
            ["Tous", "🔴 Fake seulement", "🟢 Real seulement"]
        )
    with col_f2:
        try:
            db = get_mongo()
            sources = ["Toutes"] + sorted(db.articles.distinct("source"))
        except Exception:
            sources = ["Toutes"]
        filtre_source = st.selectbox("Filtrer par source", sources)

    with col_f3:
        nb_articles = st.slider("Nombre d'articles", 10, 200, 50, 10)

    with col_f4:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Rafraîchir", use_container_width=True)

    # ── Chargement ────────────────────────────────────────────────────────────
    fake_only     = filtre_statut == "🔴 Fake seulement"
    real_only     = filtre_statut == "🟢 Real seulement"
    source_filter = None if filtre_source == "Toutes" else filtre_source

    df = fetch_articles(limit=nb_articles * 2, fake_only=fake_only, source_filter=source_filter)
    if real_only and not df.empty:
        df = df[df["is_fake"] == 0]
    df = df.head(nb_articles) if not df.empty else df

    # ── Stats rapides ─────────────────────────────────────────────────────────
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Articles affichés", len(df))
        c2.metric("Faux détectés", int(df["is_fake"].sum()))
        c3.metric("Vrais détectés", int((df["is_fake"] == 0).sum()))
        if "confidence" in df.columns:
            c4.metric("Confiance moyenne", f"{df['confidence'].mean()*100:.1f}%")
        st.markdown("---")

    # ── Distribution Confiance ────────────────────────────────────────────────
    if not df.empty and "p_fake" in df.columns:
        col_hist, col_src = st.columns([1, 1])
        with col_hist:
            st.subheader("Distribution des scores de fake")
            fig_hist = px.histogram(
                df, x="p_fake", color="is_fake",
                nbins=30, barmode="overlay",
                color_discrete_map={1: CLR_FAKE, 0: CLR_REAL},
                labels={"p_fake": "Score de faux (p_fake)", "is_fake": "Catégorie"},
                opacity=0.75
            )
            fig_hist.update_layout(
                height=250, margin=dict(t=5, b=5),
                xaxis_title="Probabilité d'être un faux article",
                yaxis_title="Nombre d'articles",
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_src:
            st.subheader("Articles par source")
            src_counts = df.groupby("source")["is_fake"].agg(
                total="count", fakes="sum"
            ).reset_index().sort_values("total", ascending=False).head(10)
            fig_src = go.Figure(data=[
                go.Bar(name="Vrais", x=src_counts["source"],
                       y=src_counts["total"] - src_counts["fakes"],
                       marker_color=CLR_REAL),
                go.Bar(name="Faux",  x=src_counts["source"],
                       y=src_counts["fakes"], marker_color=CLR_FAKE),
            ])
            fig_src.update_layout(
                barmode="stack", height=250, margin=dict(t=5, b=5),
                xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.4)
            )
            st.plotly_chart(fig_src, use_container_width=True)

    # ── Liste des articles ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📄 {len(df) if not df.empty else 0} articles classifiés")

    if df.empty:
        st.info("Aucun article correspondant aux filtres. Le pipeline est peut-être en cours de traitement...")
    else:
        for _, row in df.iterrows():
            is_fake   = row.get("is_fake", 0) == 1
            badge     = "fake" if is_fake else "real"
            badge_txt = "🔴 FAKE" if is_fake else "🟢 RÉEL"
            conf      = row.get("confidence", 0) * 100
            p_fake    = row.get("p_fake", 0) * 100
            drift_ico = "⚠️" if row.get("drift_active", False) else ""

            with st.expander(
                f"{badge_txt}  {drift_ico}  {row.get('title','(sans titre)')[:90]}",
                expanded=False
            ):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Source :** {row.get('source','?')} "
                                f"| **Langue :** {row.get('language','?')}")
                    if row.get("body"):
                        st.markdown(f"*{str(row.get('body',''))[:200]}...*")
                    if row.get("url"):
                        st.markdown(f"[🔗 Lien vers l'article]({row.get('url')})")

                with col_b:
                    st.metric("Confiance",  f"{conf:.1f}%")
                    st.metric("Score Fake", f"{p_fake:.1f}%")
                    if row.get("drift_active"):
                        st.warning(f"⚠️ Drift: {row.get('drift_score',0):.3f}")
                    processed = row.get("processed_at", "")
                    if processed:
                        st.caption(f"🕐 {str(processed)[:19]}")

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RECHERCHE & ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Recherche" in page:
    section_header("🔍", "Recherche & Analyse Full-Text (Elasticsearch)")

    st.markdown("""
    La recherche est propulsée par **Elasticsearch** avec indexation full-text
    et algorithme BM25. Recherchez des termes, des personnalités, des événements...
    """)

    col_s1, col_s2 = st.columns([4, 1])
    with col_s1:
        query = st.text_input(
            "🔎 Requête de recherche",
            placeholder="Ex: fake news, desinformation, Covid, élections...",
            label_visibility="collapsed"
        )
    with col_s2:
        nb_results = st.selectbox("Résultats", [10, 20, 50], label_visibility="collapsed")

    if query:
        with st.spinner("Recherche en cours..."):
            results = search_articles(query, size=nb_results)

        if results.empty:
            st.warning(f"Aucun résultat pour **{query}**. Essayez un autre terme.")
        else:
            # Résumé
            total_r = len(results)
            fakes_r = int(results["is_fake"].sum()) if "is_fake" in results.columns else 0
            reals_r = total_r - fakes_r

            c1, c2, c3 = st.columns(3)
            c1.metric("Résultats trouvés", total_r)
            c2.metric("🔴 Faux", fakes_r)
            c3.metric("🟢 Vrais", reals_r)

            st.markdown("---")

            # Graphique pertinence × fake
            if "p_fake" in results.columns:
                fig_scatter = px.scatter(
                    results,
                    x=results.index,
                    y="p_fake",
                    color="is_fake",
                    color_discrete_map={1: CLR_FAKE, 0: CLR_REAL},
                    size="confidence" if "confidence" in results.columns else None,
                    hover_data=["title", "source"] if "title" in results.columns else [],
                    labels={"p_fake": "Score Fake", "index": "Rang"},
                    title=f"Scores de fiabilité — Résultats pour « {query} »",
                    height=280
                )
                fig_scatter.update_layout(margin=dict(t=40, b=5), showlegend=False)
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")
            st.subheader("Résultats détaillés")
            for _, row in results.iterrows():
                is_fake   = row.get("is_fake", 0) == 1
                badge_txt = "🔴 FAKE" if is_fake else "🟢 RÉEL"
                conf      = row.get("confidence", 0) * 100

                with st.expander(f"{badge_txt}  {row.get('title', '(sans titre)')[:90]}"):
                    st.markdown(f"**Source :** {row.get('source','?')} | "
                                f"**Langue :** {row.get('language','?')} | "
                                f"**Confiance :** {conf:.1f}%")
                    if row.get("url"):
                        st.markdown(f"[🔗 Lien]({row.get('url')})")
    else:
        st.markdown("---")
        st.markdown("### 💡 Exemples de recherches")
        examples = [
            ("COVID-19", "Désinformation liée à la pandémie"),
            ("élections", "Fausses informations électorales"),
            ("fake news", "Articles sur la fake news elle-même"),
            ("Ukraine", "Désinformation géopolitique"),
            ("vaccins", "Infox sur la vaccination"),
        ]
        cols = st.columns(len(examples))
        for col, (term, desc) in zip(cols, examples):
            with col:
                st.markdown(f"""
                <div style="background:white;border-radius:8px;padding:12px;
                            text-align:center;box-shadow:0 1px 4px rgba(0,0,0,0.1);">
                    <div style="font-size:1.1rem;font-weight:600;">{term}</div>
                    <div style="font-size:0.75rem;color:#7F8C8D;margin-top:4px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DRIFT & APPRENTISSAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Drift" in page:
    section_header("📈", "Détection de Concept Drift & Apprentissage Continu")

    st.markdown("""
    Le pipeline utilise un **tri-détecteur de concept drift** basé sur River 0.21.2 :
    - **ADWIN** (poids 0.45) — dérive abrupte par fenêtre adaptative
    - **KSWIN** (poids 0.35) — changement statistique (test de Kolmogorov-Smirnov)
    - **PageHinkley** (poids 0.20) — dérive graduelle par somme cumulée
    """)

    # Paramètres de drift
    c1, c2, c3 = st.columns(3)
    c1.metric("Seuil détection", "0.5", "Score composite")
    c2.metric("Seuil confirmation", "0.8", "Score composite")
    c3.metric("Adaptation LR", "1e-5 → 5e-5", "Base → Drift")

    st.markdown("---")

    # ── Schéma du tri-détecteur ───────────────────────────────────────────────
    st.subheader("🧮 Formule du Score Composite de Drift")
    st.latex(r"S_{composite} = 0.45 \times ADWIN + 0.35 \times KSWIN + 0.20 \times PageHinkley")
    st.markdown("""
    **Décision :**
    - $S_{composite} \geq 0.5$ → **Drift détecté** → learning rate augmente à 5e-5
    - $S_{composite} \geq 0.8$ → **Drift confirmé** → mise à jour intensive du modèle
    - Après 1000 messages sans drift → **Réinitialisation** de l'alerte
    """)

    st.markdown("---")

    # ── Événements de drift ───────────────────────────────────────────────────
    drift_events = fetch_drift_events(limit=100)
    col_ev, col_stat = st.columns([2, 1])

    with col_ev:
        st.subheader(f"📋 Historique des Événements de Drift ({len(drift_events)})")
        if drift_events:
            df_drift = pd.DataFrame(drift_events)
            if "timestamp" in df_drift.columns:
                df_drift["timestamp"] = pd.to_datetime(df_drift["timestamp"])

                fig_drift = go.Figure()
                fig_drift.add_trace(go.Scatter(
                    x=df_drift["timestamp"],
                    y=df_drift.get("composite_score", [0] * len(df_drift)),
                    mode="lines+markers",
                    name="Score composite",
                    line=dict(color=CLR_DRIFT, width=2),
                    marker=dict(size=8)
                ))
                fig_drift.add_hline(y=0.5, line_dash="dash",
                                    line_color=CLR_FAKE, annotation_text="Seuil détection (0.5)")
                fig_drift.add_hline(y=0.8, line_dash="dash",
                                    line_color="#8E44AD", annotation_text="Seuil confirmation (0.8)")
                fig_drift.update_layout(
                    height=300, margin=dict(t=10, b=10),
                    yaxis=dict(title="Score", range=[0, 1]),
                    xaxis=dict(title="Temps")
                )
                st.plotly_chart(fig_drift, use_container_width=True)

            for ev in drift_events[:5]:
                confirmed = ev.get("drift_confirmed", False)
                st.markdown(f"""
                <div class="drift-alert">
                    <strong>{"⚠️ DRIFT CONFIRMÉ" if confirmed else "🔔 Drift détecté"}</strong>
                    — Score : {ev.get("composite_score", 0):.3f}<br>
                    <small>ADWIN: {ev.get("signals", {}).get("ADWIN", "?")} |
                           KSWIN: {ev.get("signals", {}).get("KSWIN", "?")} |
                           PageHinkley: {ev.get("signals", {}).get("PageHinkley", "?")}</small><br>
                    <small>🕐 {str(ev.get("timestamp",""))[:19]} |
                           LR recommandé: {ev.get("recommended_lr","?")}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Aucun drift détecté — Le modèle est stable sur les données actuelles.")
            st.markdown("""
            Le concept drift survient quand la distribution des données change significativement.
            Par exemple : une campagne de désinformation coordonnée, un événement mondial soudain,
            ou un changement de style d'écriture des fake news.
            """)

    with col_stat:
        st.subheader("📊 Statistiques")
        st.metric("Événements totaux", len(drift_events))
        confirmed = sum(1 for e in drift_events if e.get("drift_confirmed", False))
        st.metric("Drifts confirmés", confirmed)
        if drift_events:
            avg_score = np.mean([e.get("composite_score", 0) for e in drift_events])
            st.metric("Score moyen", f"{avg_score:.3f}")

        st.markdown("---")
        st.subheader("🎓 Apprentissage continu")
        st.markdown("""
        **Reservoir Sampling** : maintient 5000 exemples représentatifs pour éviter
        l'oubli catastrophique.

        **Mini-batch replay** : 8 exemples aléatoires du reservoir sont rejoués
        à chaque mise à jour.

        **Synchronisation ONNX** : tous les 100 batches, le modèle PyTorch
        est ré-exporté en ONNX INT8.
        """)

    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Infrastructure" in page:
    section_header("⚙️", "État de l'Infrastructure Docker")

    # ── Statut des services ───────────────────────────────────────────────────
    st.subheader("🐳 État des Services Docker")
    services_status = check_docker_services()

    cols = st.columns(4)
    services_info = {
        "Zookeeper":      {"icon": "🦒", "port": 2181, "role": "Coordination Kafka"},
        "Kafka":          {"icon": "📨", "port": 9092, "role": "Message Broker"},
        "MongoDB":        {"icon": "🍃", "port": 27017, "role": "Document Store"},
        "Elasticsearch":  {"icon": "🔎", "port": 9200, "role": "Moteur de recherche"},
        "FastAPI":        {"icon": "⚡", "port": 8000, "role": "API REST"},
        "Kafdrop":        {"icon": "📊", "port": 9000, "role": "Kafka UI"},
        "Grafana":        {"icon": "📈", "port": 3000, "role": "Dashboards"},
        "Streamlit":      {"icon": "🎯", "port": 8501, "role": "Ce dashboard"},
    }

    for i, (name, info) in enumerate(services_info.items()):
        col = cols[i % 4]
        is_up = services_status.get(name, False)
        status_color = "#2ECC71" if is_up else "#E74C3C"
        status_text  = "🟢 UP" if is_up else "🔴 DOWN"
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:16px;
                        margin-bottom:12px;box-shadow:0 1px 4px rgba(0,0,0,0.08);
                        border-top:3px solid {status_color};">
                <div style="font-size:1.5rem">{info['icon']}</div>
                <div style="font-weight:600;margin-top:4px">{name}</div>
                <div style="font-size:0.75rem;color:#7F8C8D">{info['role']}</div>
                <div style="font-size:0.8rem;margin-top:6px">{status_text}</div>
                <div style="font-size:0.7rem;color:#BDC3C7">port {info['port']}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Liens rapides ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔗 Liens Rapides")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.link_button("📈 Grafana", "http://localhost:3000", use_container_width=True)
    lc2.link_button("📨 Kafdrop", "http://localhost:9000", use_container_width=True)
    lc3.link_button("⚡ API Docs", "http://localhost:8000/docs", use_container_width=True)
    lc4.link_button("🔎 Elasticsearch", "http://localhost:9200", use_container_width=True)

    # ── Architecture du pipeline ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏗️ Architecture du Pipeline")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SOURCES DE DONNÉES                                    │
    │   📡 RSS Feeds (AFP, BBC, Reuters, Al Jazeera, Jeune Afrique...)         │
    │   🌐 GDELT API (Articles géopolitiques multilingues)                     │
    └──────────────────────────┬──────────────────────────────────────────────┘
                               │ scraping toutes les 60s
                               ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    APACHE KAFKA (Confluent 7.6.0)                        │
    │   📥 Topic: raw-news-stream     (6 partitions)                           │
    │   📤 Topic: classified-news     (6 partitions)                           │
    │   🔔 Topic: drift-alerts        (1 partition)                            │
    └──────────────────────────┬──────────────────────────────────────────────┘
                               │ Spark Structured Streaming
                               ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │              SPARK STREAMING (local[2]) — NLP Pipeline                   │
    │                                                                           │
    │  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐  │
    │  │  ONNX Inference │   │  Drift Detection  │   │  Online Learning    │  │
    │  │  DistilBERT INT8│   │  ADWIN+KSWIN+PH   │   │  PyTorch AdamW      │  │
    │  │  ~5-6ms/article │   │  Score composite  │   │  Reservoir Sampling │  │
    │  └─────────────────┘   └──────────────────┘   └─────────────────────┘  │
    └──────────────────────────┬──────────────────────────────────────────────┘
                               │ bulk write
                   ┌───────────┴───────────┐
                   ▼                       ▼
    ┌──────────────────────┐   ┌──────────────────────────┐
    │  MongoDB 7.0          │   │  Elasticsearch 8.14.0    │
    │  Collection: articles │   │  Index: articles          │
    │  Collection: drift    │   │  Index: drift-events      │
    └────────┬─────────────┘   └───────────┬──────────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    COUCHE PRÉSENTATION                                    │
    │  ⚡ FastAPI (port 8000) — REST API                                        │
    │  🎯 Streamlit (port 8501) — Dashboard interactif                         │
    │  📈 Grafana (port 3000) — Métriques temps réel                           │
    │  📊 Kafdrop (port 9000) — Monitoring Kafka                               │
    └──────────────────────────────────────────────────────────────────────────┘
    ```
    """)

    # ── Configuration mémoire ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Allocation Mémoire (Machine 11GB RAM)")
    mem_data = {
        "Service":    ["spark-app", "Elasticsearch", "Kafka", "MongoDB", "Kafdrop", "Grafana", "API", "Autres"],
        "Limite (MB)":[4096, 768, 640, 512, 192, 256, 256, 512],
        "Rôle":       ["ML/Spark", "Search", "Broker", "Storage", "UI", "Dashboard", "REST", "Zookeeper+RSS"]
    }
    df_mem = pd.DataFrame(mem_data)
    fig_mem = px.bar(
        df_mem, x="Service", y="Limite (MB)",
        color="Service", text="Limite (MB)",
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=280
    )
    fig_mem.update_layout(showlegend=False, margin=dict(t=5, b=5))
    fig_mem.add_hline(y=11*1024, line_dash="dash", line_color="red",
                      annotation_text="RAM totale (11GB)")
    st.plotly_chart(fig_mem, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — À PROPOS DU PROJET
# ═══════════════════════════════════════════════════════════════════════════════
elif "propos" in page:
    section_header("ℹ️", "À Propos du Projet")

    col_info, col_stats = st.columns([2, 1])

    with col_info:
        st.markdown("""
        ## Pipeline Big Data de Monitoring de la Désinformation en Temps Réel

        Ce projet constitue le sujet de mémoire de fin d'études du **Master 2 IBDIA
        (Intelligence du Big Data en Ingénierie des Affaires)** à l'**Université Catholique
        de l'Afrique de l'Ouest — Unité Universitaire du Togo (UCAO-UUT)**.

        ### 🎯 Objectifs

        1. **Collecte en temps réel** d'articles d'actualité via RSS (AFP, BBC, Reuters,
           Al Jazeera, Jeune Afrique...) et l'API GDELT
        2. **Classification automatique** fake/réel par un modèle Continual-DistilBERT
           (ONNX INT8 — multilingue)
        3. **Détection de concept drift** pour adapter le modèle aux nouvelles formes
           de désinformation
        4. **Apprentissage continu** avec reservoir sampling pour éviter l'oubli
           catastrophique
        5. **Visualisation temps réel** via ce dashboard Streamlit, Grafana et une API REST

        ### 🔬 Innovation Technique

        - **Continual Learning** : Le modèle s'adapte en continu sans être réentraîné
          depuis zéro
        - **Tri-détecteur hybride** : Combinaison ADWIN + KSWIN + PageHinkley avec
          score composite pondéré
        - **Inférence ultra-rapide** : ONNX INT8 quantifié (75% de compression, 5-6ms/article)
        - **Architecture cloud-ready** : Entièrement dockerisée, reproducible

        ### 📊 Stack Technologique
        """)

        tech_data = {
            "Composant": ["Apache Kafka", "Apache Spark", "DistilBERT", "ONNX Runtime",
                          "River", "MongoDB", "Elasticsearch", "FastAPI", "Streamlit"],
            "Version": ["3.7 (Confluent 7.6)", "3.5.3", "multilingual-cased",
                        "1.19.0", "0.21.2", "7.0", "8.14.0", "0.113", "1.38"],
            "Rôle": ["Message Broker", "Streaming Engine", "Modèle NLP",
                     "Inférence rapide", "Concept Drift", "Document Store",
                     "Full-Text Search", "API REST", "Dashboard interactif"]
        }
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    with col_stats:
        st.markdown("### 👤 Auteur")
        st.markdown("""
        **KOMOSSI Sosso**
        Étudiant Master 2 IBDIA
        UCAO-UUT, Lomé, Togo
        Année 2025-2026

        ---
        **Encadrants :**
        M. TCHANTCHO Leri
        M. BABA Kpatcha

        ---
        """)

        st.markdown("### 📈 Performances Modèle")
        perf = {
            "Métrique": ["F1-Score", "AUC-ROC", "Latence ONNX", "Compression"],
            "Valeur": ["98.49%", "99.89%", "~5-6 ms", "75%"]
        }
        st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)

        st.markdown("### 📚 Datasets")
        st.markdown("""
        - **FakeNewsNet** (PolitiFact + GossipCop)
        - **Fakeddit** (Reddit multimodal)
        - **RSS temps réel** (12 sources)
        - **GDELT** (géopolitique multilingue)
        """)

    # ── Endpoints API ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚡ Endpoints API REST")
    endpoints = [
        ("GET", "/health",                        "Santé des services (MongoDB + ES)"),
        ("GET", "/api/v1/stats",                  "Statistiques globales du pipeline"),
        ("GET", "/api/v1/articles/recent",        "Derniers articles classifiés"),
        ("GET", "/api/v1/articles/search?q=...",  "Recherche full-text Elasticsearch"),
        ("GET", "/api/v1/articles/virality",      "Tendance horaire du taux de faux"),
        ("GET", "/api/v1/drift/events",           "Historique des alertes de drift"),
        ("GET", "/api/v1/drift/stats",            "Statistiques agrégées sur le drift"),
    ]
    df_ep = pd.DataFrame(endpoints, columns=["Méthode", "Endpoint", "Description"])
    st.dataframe(df_ep, use_container_width=True, hide_index=True,
                 column_config={"Méthode": st.column_config.TextColumn(width="small")})

    st.markdown("---")
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    col_l1.link_button("📖 API Documentation", "http://localhost:8000/docs", use_container_width=True)
    col_l2.link_button("📈 Grafana", "http://localhost:3000", use_container_width=True)
    col_l3.link_button("📨 Kafdrop", "http://localhost:9000", use_container_width=True)
    col_l4.link_button("🔎 Elasticsearch", "http://localhost:9200", use_container_width=True)
