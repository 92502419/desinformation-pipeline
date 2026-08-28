# scripts/preprocess_data.py
# Fusionne les datasets bruts (ISOT, WELFake, FakeNewsNet, LIAR, + sous-corpus
# africain multilingue optionnel) et produit les CSV train/val/test prêts à l'emploi.
#
# Usage (depuis la racine du projet, avec venv_main activé) :
#   python scripts/preprocess_data.py

import pandas as pd, os, re, sys
from sklearn.model_selection import train_test_split

if not os.path.exists('scripts/preprocess_data.py'):
    print('ERREUR : Lancer depuis la racine du projet')
    print('  cd ~/desinformation-pipeline')
    sys.exit(1)

BASE = 'data/raw'
OUT = 'data/processed'
os.makedirs(f'{OUT}/train', exist_ok=True)
os.makedirs(f'{OUT}/val', exist_ok=True)
os.makedirs(f'{OUT}/test', exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:2000]


dfs = []

# ── 1. WELFAKE ───────────────────────────────────────────
welfake_path = f'{BASE}/welfake/WELFake_Dataset.csv'
if os.path.exists(welfake_path):
    print('Chargement WELFake...')
    df = pd.read_csv(welfake_path)
    df = df.rename(columns={'text': 'body'})
    df['source'] = 'welfake'
    df = df[['title', 'body', 'label', 'source']].dropna(subset=['title'])
    df['label'] = df['label'].astype(int)  # 0=réel, 1=fake
    dfs.append(df)
    print(f'  WELFake : {len(df)} exemples')
else:
    print(f'  WELFake : SKIP (fichier introuvable : {welfake_path})')

# ── 2. ISOT ──────────────────────────────────────────────
isot_fake = f'{BASE}/isot/Fake.csv'
isot_real = f'{BASE}/isot/True.csv'
if os.path.exists(isot_fake) and os.path.exists(isot_real):
    print('Chargement ISOT...')
    fake = pd.read_csv(isot_fake).assign(label=1, source='isot')
    real = pd.read_csv(isot_real).assign(label=0, source='isot')
    isot = pd.concat([fake, real]).rename(columns={'text': 'body'})
    isot = isot[['title', 'body', 'label', 'source']].dropna(subset=['title'])
    dfs.append(isot)
    print(f'  ISOT : {len(isot)} exemples')
else:
    print(f'  ISOT : SKIP (Fake.csv ou True.csv introuvable dans {BASE}/isot/)')

# ── 3. LIAR ──────────────────────────────────────────────
print('Chargement LIAR...')
REAL_LABELS = {'true', 'mostly-true', 'half-true'}
liar_parts = []
for split in ['train', 'valid', 'test']:
    fp = f'{BASE}/liar/{split}.tsv'
    if not os.path.exists(fp):
        print(f'  LIAR {split} : non trouvé ({fp})')
        continue
    try:
        tmp = pd.read_csv(fp, sep='\t', header=None, on_bad_lines='skip',
                           names=['id', 'label', 'statement', 'subjects', 'speaker', 'job', 'state',
                                  'party', 'bc', 'fc', 'hc', 'fc2', 'pants', 'context'])
        tmp['label'] = tmp['label'].apply(lambda x: 0 if str(x).strip() in REAL_LABELS else 1)
        tmp['title'] = tmp['statement'].fillna('')
        tmp['body'] = tmp['context'].fillna('')
        tmp['source'] = 'liar'
        liar_parts.append(tmp[['title', 'body', 'label', 'source']])
    except Exception as e:
        print(f'  LIAR {split} : erreur ({e})')
if liar_parts:
    liar_df = pd.concat(liar_parts)
    dfs.append(liar_df)
    print(f'  LIAR : {len(liar_df)} exemples')

# ── 4. FAKENEWSNET — Lecture directe des CSV locaux ───────
print('Chargement FakeNewsNet (CSV locaux)...')
fnn_dir = f'{BASE}/fakenewsnet/dataset'
fnn_parts = []
fnn_files = {
    'gossipcop_fake.csv': 1,
    'gossipcop_real.csv': 0,
    'politifact_fake.csv': 1,
    'politifact_real.csv': 0,
}
for fname, lbl in fnn_files.items():
    fp = os.path.join(fnn_dir, fname)
    if not os.path.exists(fp):
        print(f'  FakeNewsNet {fname} : non trouvé — skip')
        continue
    try:
        tmp = pd.read_csv(fp, on_bad_lines='skip')
        title_col = next((c for c in tmp.columns if 'title' in c.lower()), None)
        body_col = next((c for c in tmp.columns if c.lower() in ['text', 'body', 'content', 'news_url']), None)
        if title_col is None:
            print(f'  FakeNewsNet {fname} : pas de colonne title — skip')
            continue
        tmp = tmp.rename(columns={title_col: 'title'})
        tmp['body'] = tmp[body_col].fillna('') if body_col else ''
        tmp['label'] = lbl
        tmp['source'] = 'fakenewsnet'
        fnn_parts.append(tmp[['title', 'body', 'label', 'source']].dropna(subset=['title']))
        print(f'  {fname} : {len(tmp)} exemples')
    except Exception as e:
        print(f'  FakeNewsNet {fname} : erreur ({e})')
if fnn_parts:
    dfs.append(pd.concat(fnn_parts))
    print(f'  FakeNewsNet total : {sum(len(p) for p in fnn_parts)} exemples')
else:
    print('  FakeNewsNet : aucun CSV trouvé dans', fnn_dir)

# ── 5. Sous-corpus africain multilingue (v2.4, optionnel) ─
africa_path = f'{BASE}/africa_news/africa_news.csv'
if os.path.exists(africa_path):
    print('Chargement sous-corpus africain (MasakhaNEWS + RSS)...')
    africa = pd.read_csv(africa_path)
    africa = africa[['title', 'body', 'label', 'source']].dropna(subset=['title'])
    dfs.append(africa)
    print(f'  Afrique : {len(africa)} exemples')
else:
    print(f'  Afrique : SKIP (lancer scripts/download_african_datasets.py d\'abord)')

# ── FUSION & NETTOYAGE ───────────────────────────────────
if not dfs:
    print('ERREUR : Aucun dataset chargé ! Vérifier les chemins.')
    sys.exit(1)

print('\nFusion et nettoyage...')
corpus = pd.concat(dfs, ignore_index=True)
print(f'Avant nettoyage : {len(corpus)} exemples')

corpus['title'] = corpus['title'].apply(clean_text)
corpus['body'] = corpus['body'].apply(clean_text)

# Suppression des titres vides ou trop courts
corpus = corpus[corpus['title'].str.len() > 5]

# Déduplication sur le titre
n_before = len(corpus)
corpus = corpus.drop_duplicates(subset=['title'])
print(f'Doublons supprimés : {n_before - len(corpus)}')
print(f'Corpus final : {len(corpus)} exemples')
print(corpus[['label', 'source']].value_counts().to_string())

# ── ÉQUILIBRAGE 50/50 PUIS SPLIT 60/20/20 (temporel-ish via seed fixe) ─
n_min = corpus['label'].value_counts().min()
corpus = pd.concat([g.sample(n=n_min, random_state=42) for _, g in corpus.groupby('label')],
                    ignore_index=True)
train_val, test = train_test_split(
    corpus, test_size=0.20, stratify=corpus['label'], random_state=42)
train, val = train_test_split(
    train_val, test_size=0.25, stratify=train_val['label'], random_state=42)

train.to_csv(f'{OUT}/train/train.csv', index=False)
val.to_csv(f'{OUT}/val/val.csv', index=False)
test.to_csv(f'{OUT}/test/test.csv', index=False)

# Rapport qualité (traçabilité pour le mémoire)
report = {
    'total_before_dedup': int(n_before),
    'total_after_dedup': int(len(corpus) + 0),  # avant équilibrage, recalculé ci-dessous
    'duplicates_removed': int(n_before - len(corpus)),
    'n_train': int(len(train)),
    'n_val': int(len(val)),
    'n_test': int(len(test)),
    'sources': corpus['source'].value_counts().to_dict(),
}
import json
with open(f'{OUT}/preprocessing_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f'\nTrain : {len(train)} | Val : {len(val)} | Test : {len(test)}')
print('Prétraitement terminé !')
print(f'Fichiers générés dans {OUT}/ (+ preprocessing_report.json)')
