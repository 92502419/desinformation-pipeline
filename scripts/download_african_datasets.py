#!/usr/bin/env python3
# scripts/download_african_datasets.py — Enrichissement africain multilingue (v2.4)
# Usage (racine du projet, venv_main activé) : python scripts/download_african_datasets.py
#
# Construit data/raw/africa_news/africa_news.csv à partir de :
#   - MasakhaNEWS (HuggingFace `masakhane/masakhanews`) : 2 742 articles réels
#     couvrant 11 langues africaines (amh, hau, ibo, lin, orm, pcm, run, sna, som, swa, tir, yor)
#     -> tous labellisés réels (label=0), ce sont des dépêches de presse vérifiées.
#   - Un petit lot d'articles complémentaires récents via Google News RSS
#     (sources africaines) pour diversifier titres/domaines récents.
#
# NB : Africa Check est inaccessible depuis cet environnement (blocage Cloudflare 403) —
# limite documentée : peu d'exemples "fake" authentiquement africains (~77), à enrichir
# dans des travaux futurs.

import os, sys, csv, time
import pandas as pd

if not os.path.exists('scripts/download_african_datasets.py'):
    print('ERREUR : Lancer depuis la racine du projet (cd ~/desinformation-pipeline)')
    sys.exit(1)

OUT_DIR = 'data/raw/africa_news'
OUT_CSV = f'{OUT_DIR}/africa_news.csv'
os.makedirs(OUT_DIR, exist_ok=True)

MASAKHANEWS_LANGS = ['amh', 'hau', 'ibo', 'lin', 'orm', 'pcm', 'run', 'sna', 'som', 'swa', 'tir', 'yor']

rows = []

# ── 1. MasakhaNEWS (HuggingFace) ────────────────────────────────
try:
    from datasets import load_dataset
    print(f'Téléchargement MasakhaNEWS ({len(MASAKHANEWS_LANGS)} langues)...')
    for lang in MASAKHANEWS_LANGS:
        try:
            ds = load_dataset('masakhane/masakhanews', lang, trust_remote_code=True)
            for split in ds.keys():
                df = ds[split].to_pandas()
                text_col = 'headline' if 'headline' in df.columns else df.columns[0]
                for _, r in df.iterrows():
                    rows.append({
                        'title': str(r.get(text_col, ''))[:300],
                        'body': str(r.get('text', ''))[:1000] if 'text' in df.columns else '',
                        'label': 0,  # articles de presse vérifiés → réel
                        'source': f'masakhanews_{lang}',
                    })
            print(f'  {lang} : OK ({len([r for r in rows if r["source"] == f"masakhanews_{lang}"])} ex.)')
        except Exception as e:
            print(f'  {lang} : SKIP ({e})')
except ImportError:
    print('ATTENTION : `datasets` non installé — MasakhaNEWS ignoré (pip install datasets)')

print(f'MasakhaNEWS total : {len(rows)} exemples')

# ── 2. Complément Google News RSS (sources africaines) ──────────
AFRICA_RSS_QUERIES = [
    'Afrique+de+l%27Ouest', 'AFP+Afrique', 'RFI+Afrique', 'Jeune+Afrique',
    'Al+Jazeera+Africa', 'France24+Afrique', 'VOA+Afrique',
]
try:
    import feedparser
    n_before = len(rows)
    for q in AFRICA_RSS_QUERIES:
        url = f'https://news.google.com/rss/search?q={q}&hl=fr&gl=FR&ceid=FR:fr'
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                rows.append({
                    'title': getattr(entry, 'title', '')[:300],
                    'body': getattr(entry, 'summary', '')[:1000],
                    'label': 0,
                    'source': 'google_news_africa_rss',
                })
        except Exception as e:
            print(f'  RSS {q} : SKIP ({e})')
        time.sleep(0.5)
    print(f'Google News RSS Afrique : {len(rows) - n_before} exemples ajoutés')
except ImportError:
    print('ATTENTION : `feedparser` non installé — complément RSS ignoré (pip install feedparser)')

if not rows:
    print('ERREUR : Aucun exemple africain collecté (vérifier connexion réseau / accès HuggingFace).')
    print('Le pipeline peut continuer sans ce sous-corpus (moins bonne couverture africaine).')
    sys.exit(0)

df = pd.DataFrame(rows).drop_duplicates(subset=['title'])
df = df[df['title'].str.len() > 5]
df.to_csv(OUT_CSV, index=False)
print(f'\nCorpus africain final : {len(df)} exemples -> {OUT_CSV}')
print(df['source'].value_counts().to_string())
print('Prochaine étape : python scripts/preprocess_data.py')
