# scripts/preprocess_data.py
# Fusionne les 4 datasets et produit train/val/test CSV prêts à l'emploi
# Usage (depuis la racine du projet, avec venv_main activé) :
#   python scripts/preprocess_data.py


import pandas as pd, os, re, sys
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Vérification qu'on est bien à la racine du projet
if not os.path.exists('scripts/preprocess_data.py'):
    print('ERREUR : Lancer depuis la racine du projet')
    print('  cd ~/desinformation-pipeline')
    sys.exit(1)


BASE = 'data/raw'
OUT  = 'data/processed'
os.makedirs(f'{OUT}/train', exist_ok=True)
os.makedirs(f'{OUT}/val',   exist_ok=True)
os.makedirs(f'{OUT}/test',  exist_ok=True)


def clean_text(text: str) -> str:
    if not isinstance(text, str): return ''
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;',  '<', text)
    text = re.sub(r'&gt;',  '>', text)
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
    df = df[['title','body','label','source']].dropna(subset=['title'])
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
    isot = isot[['title','body','label','source']].dropna(subset=['title'])
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
            names=['id','label','statement','subjects','speaker','job','state',
                   'party','bc','fc','hc','fc2','pants','context'])
        tmp['label'] = tmp['label'].apply(lambda x: 0 if str(x).strip() in REAL_LABELS else 1)
        tmp['title'] = tmp['statement'].fillna('')
        tmp['body']  = tmp['context'].fillna('')
        tmp['source'] = 'liar'
        liar_parts.append(tmp[['title','body','label','source']])
    except Exception as e:
        print(f'  LIAR {split} : erreur ({e})')
if liar_parts:
    liar_df = pd.concat(liar_parts)
    dfs.append(liar_df)
    print(f'  LIAR : {len(liar_df)} exemples')


# ── 4. FAKENEWSNET — Lecture directe des CSV locaux ───────
# Structure réelle du dépôt KaiDMML/FakeNewsNet :
#   data/raw/fakenewsnet/dataset/gossipcop_fake.csv
#   data/raw/fakenewsnet/dataset/gossipcop_real.csv
#   data/raw/fakenewsnet/dataset/politifact_fake.csv
#   data/raw/fakenewsnet/dataset/politifact_real.csv
print('Chargement FakeNewsNet (CSV locaux)...')
fnn_dir = f'{BASE}/fakenewsnet/dataset'
fnn_parts = []
fnn_files = {
    'gossipcop_fake.csv':   1,   # label 1 = fake
    'gossipcop_real.csv':   0,   # label 0 = réel
    'politifact_fake.csv':  1,
    'politifact_real.csv':  0,
}
for fname, lbl in fnn_files.items():
    fp = os.path.join(fnn_dir, fname)
    if not os.path.exists(fp):
        print(f'  FakeNewsNet {fname} : non trouvé — skip')
        continue
    try:
        tmp = pd.read_csv(fp, on_bad_lines='skip')
        # Les colonnes varient selon le fichier — on cherche title et body
        title_col = next((c for c in tmp.columns if 'title' in c.lower()), None)
        body_col  = next((c for c in tmp.columns if c.lower() in ['text','body','content','news_url']), None)
        if title_col is None:
            print(f'  FakeNewsNet {fname} : pas de colonne title — skip')
            continue
        tmp = tmp.rename(columns={title_col: 'title'})
        tmp['body']   = tmp[body_col].fillna('') if body_col else ''
        tmp['label']  = lbl
        tmp['source'] = 'fakenewsnet'
        fnn_parts.append(tmp[['title','body','label','source']].dropna(subset=['title']))
        print(f'  {fname} : {len(tmp)} exemples')
    except Exception as e:
        print(f'  FakeNewsNet {fname} : erreur ({e})')
if fnn_parts:
    dfs.append(pd.concat(fnn_parts))
    print(f'  FakeNewsNet total : {sum(len(p) for p in fnn_parts)} exemples')
else:
    print('  FakeNewsNet : aucun CSV trouvé dans', fnn_dir)


# ── FUSION & NETTOYAGE ───────────────────────────────────
if not dfs:
    print('ERREUR : Aucun dataset chargé ! Vérifier les chemins.')
    sys.exit(1)


print('\nFusion et nettoyage...')
corpus = pd.concat(dfs, ignore_index=True)
print(f'Avant nettoyage : {len(corpus)} exemples')

corpus['title'] = corpus['title'].apply(clean_text)
corpus['body']  = corpus['body'].apply(clean_text)
corpus = corpus[corpus['title'].str.len() > 5]
n_before = len(corpus)
corpus = corpus.drop_duplicates(subset=['title'])
print(f'Doublons supprimés : {n_before - len(corpus)}')
print(corpus[['label','source']].value_counts().to_string())


# ── ÉQUILIBRAGE DES CLASSES (50 % fake / 50 % réel) ─────────
# Évite que le modèle apprenne à détecter le fake mieux que le vrai.
# Sous-échantillonnage de la classe majoritaire vers la plus petite.
fake_df   = corpus[corpus['label'] == 1]
real_df   = corpus[corpus['label'] == 0]
min_count = min(len(fake_df), len(real_df))
corpus_balanced = pd.concat([
    fake_df.sample(min_count, random_state=42),
    real_df.sample(min_count, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f'\nCorpus équilibré : {len(corpus_balanced)} exemples '
      f'({min_count} fake + {min_count} réel = 50 % / 50 %)')


# ── SPLIT 60 / 20 / 20 ──────────────────────────────────────
# 60 % entraînement / 20 % validation / 20 % test
# Split stratifié : conserve le ratio 50/50 dans chaque sous-ensemble
train_tmp, test = train_test_split(
    corpus_balanced, test_size=0.20,
    stratify=corpus_balanced['label'], random_state=42)
train, val = train_test_split(
    train_tmp, test_size=0.25,
    stratify=train_tmp['label'], random_state=42)
# 0.25 × 80 % = 20 % du total → split final 60 / 20 / 20


train.to_csv(f'{OUT}/train/train.csv', index=False)
val.to_csv(  f'{OUT}/val/val.csv',     index=False)
test.to_csv( f'{OUT}/test/test.csv',   index=False)


print(f'\nTrain : {len(train)} ({len(train[train.label==1])} fake / {len(train[train.label==0])} réel)')
print(f'Val   : {len(val)}   ({len(val[val.label==1])} fake / {len(val[val.label==0])} réel)')
print(f'Test  : {len(test)}  ({len(test[test.label==1])} fake / {len(test[test.label==0])} réel)')
print('Prétraitement terminé !')
print(f'Fichiers générés dans {OUT}/')

