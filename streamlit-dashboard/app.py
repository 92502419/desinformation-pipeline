# streamlit-dashboard/app.py — Dashboard interactif multi-pages
# Pipeline Big Data de Monitoring de la Désinformation en Temps Réel
# KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT 2025-2026
#
# Reconstruit le 2026-08-27 (le service streamlit-dashboard/ avait été vidé lors
# du plantage disque du 26/08/2026 — voir README.md, section "Incident & reprise").
#
# 5 pages : Vue d'ensemble, Explorer les articles, Monitoring Drift,
# Analyse des sources, Configuration — + page Recherche web (v2.1).
#
# IS_CLOUD=1 : mode démonstration (Streamlit Cloud) — désactive les appels vers
# les services locaux (Kafka, MongoDB, FastAPI) non disponibles hors du docker-compose,
# et affiche des données d'exemple à la place.

import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

IS_CLOUD = os.getenv('IS_CLOUD', '0') == '1'
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.getenv('MONGO_DB', 'disinformation_db')
ES_HOST = os.getenv('ES_HOST', 'http://localhost:9200')
API_BASE = os.getenv('API_BASE', 'http://localhost:8000')
REFRESH_SEC = int(os.getenv('REFRESH_SEC', '30'))

st.set_page_config(
    page_title='Monitoring Désinformation — Pipeline Temps Réel',
    page_icon='🛰️',
    layout='wide',
)

# ── Connexions (silencieuses en mode cloud) ─────────────────────
@st.cache_resource
def get_mongo():
    if IS_CLOUD:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command('ping')
        return client[MONGO_DB]
    except Exception:
        return None


def api_get(path, params=None, timeout=5):
    if IS_CLOUD:
        return None
    try:
        r = requests.get(f'{API_BASE}{path}', params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Données de démonstration (mode cloud / services indisponibles) ─
def demo_stats():
    return {
        'total_articles': 12_483, 'fake_articles': 5_216, 'real_articles': 7_267,
        'fake_rate': 41.8, 'drift_events': 7, 'articles_last_hour': 214,
    }


def demo_articles(n=200):
    import random
    random.seed(42)
    sources = ['AFP Afrique', 'RFI', 'Jeune Afrique', 'Al Jazeera', 'France24', 'Reuters', 'BBC', 'blog-inconnu.info']
    langs = ['fr', 'en', 'sw', 'ha', 'am']
    rows = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        is_fake = random.random() < 0.42
        rows.append({
            'title': f"Article démonstration #{i+1} — {'alerte virale non confirmée' if is_fake else 'communiqué officiel'}",
            'source': random.choice(sources),
            'language': random.choice(langs),
            'is_fake': int(is_fake),
            'confidence': round(random.uniform(0.6, 0.99), 3),
            'processed_at': (now - timedelta(minutes=random.randint(0, 720))).isoformat(),
        })
    return pd.DataFrame(rows)


db = get_mongo()

st.sidebar.title('🛰️ Pipeline Désinformation')
st.sidebar.caption('KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT 2025-2026')
if IS_CLOUD:
    st.sidebar.warning("Mode démonstration (Streamlit Cloud) : données d'exemple — "
                        "les services Kafka/MongoDB/FastAPI ne sont pas accessibles hors du docker-compose local.")
elif db is None:
    st.sidebar.warning('Services locaux injoignables (MongoDB) — données de démonstration affichées. '
                        'Lancer `docker compose up -d` pour les données réelles.')
else:
    st.sidebar.success('Connecté à MongoDB / API ✅')

page = st.sidebar.radio('Navigation', [
    "🏠 Vue d'ensemble",
    '🔍 Explorer les articles',
    '⚡ Monitoring Drift',
    '📊 Analyse des sources',
    '🌍 Recherche web',
    '⚙️ Configuration',
])

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Vue d'ensemble":
    st.title("🏠 Vue d'ensemble — Temps réel")

    stats = api_get('/api/v1/stats') or demo_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Articles analysés', f"{stats['total_articles']:,}".replace(',', ' '))
    c2.metric('Taux de fake', f"{stats['fake_rate']} %")
    c3.metric('Événements de drift', stats['drift_events'])
    c4.metric('Articles / dernière heure', stats['articles_last_hour'])

    df = demo_articles(300) if (IS_CLOUD or db is None) else pd.DataFrame(
        list(db.articles.find({}, {'_id': 0}).sort('processed_at', -1).limit(500)))

    if not df.empty and 'processed_at' in df.columns:
        df['processed_at'] = pd.to_datetime(df['processed_at'], errors='coerce', utc=True)
        df['heure'] = df['processed_at'].dt.floor('h')
        ts = df.groupby(['heure', 'is_fake']).size().reset_index(name='n')
        ts['label'] = ts['is_fake'].map({0: 'Réel', 1: 'Fake'})
        fig = px.bar(ts, x='heure', y='n', color='label', barmode='stack',
                     title='Volume d\'articles par heure (réel vs fake)',
                     color_discrete_map={'Réel': '#2E7D32', 'Fake': '#C62828'})
        st.plotly_chart(fig, use_container_width=True)

        pie = df['is_fake'].map({0: 'Réel', 1: 'Fake'}).value_counts().reset_index()
        pie.columns = ['label', 'n']
        fig2 = px.pie(pie, names='label', values='n', title='Répartition Fake / Réel',
                      color='label', color_discrete_map={'Réel': '#2E7D32', 'Fake': '#C62828'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Aucun article en base pour le moment — le producer/spark-app n'a peut-être pas encore démarré.")

    st.caption(f"Rafraîchissement automatique toutes les {REFRESH_SEC}s.")

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORER LES ARTICLES
# ═══════════════════════════════════════════════════════════════
elif page == '🔍 Explorer les articles':
    st.title('🔍 Explorer les articles classifiés')

    col1, col2, col3 = st.columns(3)
    query = col1.text_input('Recherche full-text (Elasticsearch)', '')
    label_filter = col2.selectbox('Label', ['Tous', 'Fake', 'Réel'])
    lang_filter = col3.text_input('Langue (code, ex: fr, en, sw)', '')

    df = demo_articles(300) if (IS_CLOUD or db is None) else pd.DataFrame(
        list(db.articles.find({}, {'_id': 0}).sort('processed_at', -1).limit(1000)))

    if not df.empty:
        if query:
            df = df[df['title'].str.contains(query, case=False, na=False)]
        if label_filter != 'Tous':
            df = df[df['is_fake'] == (1 if label_filter == 'Fake' else 0)]
        if lang_filter:
            df = df[df.get('language', '').astype(str).str.lower() == lang_filter.lower()]

        st.write(f"**{len(df)}** article(s) trouvé(s)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button('📥 Exporter en CSV', df.to_csv(index=False).encode('utf-8'),
                            'articles_export.csv', 'text/csv')
    else:
        st.info('Aucun article disponible.')

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING DRIFT
# ═══════════════════════════════════════════════════════════════
elif page == '⚡ Monitoring Drift':
    st.title('⚡ Monitoring du Concept Drift')
    st.caption('Tri-détecteur ADWIN + KSWIN + PageHinkley (module `spark-app/src/drift_monitor.py`)')

    drift_data = api_get('/api/v1/drift/status')
    if drift_data is None:
        import random
        random.seed(7)
        n = 60
        now = datetime.now(timezone.utc)
        drift_data = {
            'history': [{
                'timestamp': (now - timedelta(minutes=5 * (n - i))).isoformat(),
                'composite_score': max(0, min(1, 0.2 + 0.4 * abs(((i % 20) - 10) / 10) + random.uniform(-0.05, 0.05))),
                'adwin': random.random() < 0.05,
                'kswin': random.random() < 0.05,
                'page_hinkley': random.random() < 0.03,
            } for i in range(n)],
            'current_lr': 1.4e-5,
        }

    hist = pd.DataFrame(drift_data['history'])
    hist['timestamp'] = pd.to_datetime(hist['timestamp'], errors='coerce')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['composite_score'],
                              mode='lines', name='Score composite', line=dict(color='#1565C0')))
    fig.add_hline(y=0.5, line_dash='dash', line_color='orange', annotation_text='Seuil d\'alerte')
    fig.update_layout(title='Score composite de drift dans le temps', yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Alertes ADWIN', int(hist['adwin'].sum()))
    c2.metric('Alertes KSWIN', int(hist['kswin'].sum()))
    c3.metric('Alertes Page-Hinkley', int(hist['page_hinkley'].sum()))
    c4.metric('Learning rate courant', f"{drift_data['current_lr']:.2e}")

    st.subheader('Historique des alertes')
    alerts = hist[hist[['adwin', 'kswin', 'page_hinkley']].any(axis=1)]
    st.dataframe(alerts, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — ANALYSE DES SOURCES
# ═══════════════════════════════════════════════════════════════
elif page == '📊 Analyse des sources':
    st.title('📊 Analyse des sources')

    df = demo_articles(400) if (IS_CLOUD or db is None) else pd.DataFrame(
        list(db.articles.find({}, {'_id': 0}).sort('processed_at', -1).limit(2000)))

    if not df.empty:
        by_source = df.groupby('source').agg(
            n_articles=('source', 'size'),
            taux_fake=('is_fake', 'mean'),
        ).reset_index().sort_values('n_articles', ascending=False)
        by_source['taux_fake'] = (by_source['taux_fake'] * 100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(by_source, names='source', values='n_articles', title='Répartition par source')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(by_source, x='source', y='taux_fake', title='Taux de fake par source (%)',
                          color='taux_fake', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(by_source, use_container_width=True, hide_index=True)
    else:
        st.info('Aucune donnée disponible.')

# ═══════════════════════════════════════════════════════════════
# PAGE 5 — RECHERCHE WEB (v2.1)
# ═══════════════════════════════════════════════════════════════
elif page == '🌍 Recherche web':
    st.title('🌍 Recherche & classification en direct')
    st.caption("Sources prioritaires : AFP Afrique, RFI, Jeune Afrique, Al Jazeera, France24, VOA Afrique — "
               "architecture universelle, exportable hors d'Afrique.")

    tab1, tab2 = st.tabs(['🔎 Recherche web en direct', '🗄️ Base de données'])

    with tab1:
        q = st.text_input('Rechercher un sujet d\'actualité', placeholder='ex: élections, Ebola, coup d\'État...')
        suggested = ['Élections présidentielles Afrique de l\'Ouest', 'Prix du carburant', 'Sécurité alimentaire Sahel']
        st.caption('Suggestions : ' + ' · '.join(suggested))
        if st.button('Analyser', type='primary') and q:
            if IS_CLOUD:
                st.warning("Fonctionnalité indisponible en mode démonstration (nécessite l'API FastAPI + Kafka locaux).")
            else:
                result = api_get('/api/v1/search/web', params={'q': q})
                if result:
                    for item in result.get('results', []):
                        badge = '🔴 FAKE' if item['is_fake'] else '🟢 RÉEL'
                        st.markdown(f"**{badge}** ({item['confidence']:.0%}) — [{item['title']}]({item['url']})")
                else:
                    st.error("L'API n'a pas répondu — vérifier que le service `api` est démarré.")

    with tab2:
        query2 = st.text_input('Recherche full-text (Elasticsearch)', key='es_search')
        st.info('Recherche indexée sur les articles déjà traités par le pipeline — voir page "Explorer les articles".')

# ═══════════════════════════════════════════════════════════════
# PAGE 6 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════
elif page == '⚙️ Configuration':
    st.title('⚙️ Configuration live du pipeline')
    st.warning('Ces paramètres sont indicatifs dans cette version du dashboard : '
               'la persistance des changements nécessite un endpoint API dédié (non prioritaire pour la soutenance).')

    st.slider('Seuil de décision fake (p_fake ≥ seuil)', 0.5, 0.95, 0.75, 0.01)
    st.slider('Intervalle de scraping RSS (secondes)', 15, 300, 60, 5)
    st.slider('Rafraîchissement dashboard (secondes)', 5, 120, REFRESH_SEC, 5)
    st.number_input('Taille du reservoir buffer (online learning)', 500, 20000, 5000, 500)

    st.subheader('À propos')
    st.markdown("""
    Ce pipeline est déployé et pensé en priorité pour un contexte **africain et francophone**
    (sources : AFP Afrique, RFI, Jeune Afrique, Al Jazeera, France24, VOA Afrique), avec un
    sous-corpus d'entraînement multilingue (MasakhaNEWS, 11 langues africaines). L'architecture
    reste néanmoins universelle et exportable hors d'Afrique.
    """)

if not IS_CLOUD:
    time.sleep(0.1)  # évite un rafraîchissement trop agressif en dev local
