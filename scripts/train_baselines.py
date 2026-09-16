#!/usr/bin/env python3
# scripts/train_baselines.py — Baselines statiques pour comparaison avec Continual-DistilBERT
#
# Répond à un écart identifié dans reports/CONFORMITE_PROPOSITION.md (H2 / Tableau 4.1) :
# aucune baseline classique n'avait été ré-entraînée après l'incident du 26/08/2026.
# Ce script entraîne deux baselines statiques, sur le MÊME split que le modèle
# Transformer (data/processed/{train,test}.csv), et les évalue sur le même jeu de
# test tenu à l'écart :
#   - TF-IDF (1-2-grammes) + Régression Logistique
#   - TF-IDF (1-2-grammes) + Linear SVM (SGDClassifier, loss='hinge')
#
# Aucune donnée inventée : F1-macro, précision, rappel, AUC, matrice de confusion
# sont calculés sur data/processed/test/test.csv, à comparer directement à
# reports/metrics_summary.json (Continual-DistilBERT, F1-macro 0,9370).
#
# Sortie : reports/baselines_statiques.json + reports/baselines_statiques.md
#
# Usage : python scripts/train_baselines.py

import os, sys, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, 'data/processed/train/train.csv')
TEST_CSV  = os.path.join(ROOT, 'data/processed/test/test.csv')
REPORTS   = os.path.join(ROOT, 'reports')


def load_text(df):
    title = df['title'].fillna('').astype(str)
    body = df['body'].fillna('').astype(str)
    return (title + ' ' + body).str.slice(0, 2000)


def evaluate(name, model, Xtr, ytr, Xte, yte, t0):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  roc_auc_score, average_precision_score,
                                  confusion_matrix)
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    if hasattr(model, 'predict_proba'):
        score = model.predict_proba(Xte)[:, 1]
    elif hasattr(model, 'decision_function'):
        raw = model.decision_function(Xte)
        score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    else:
        score = yp.astype(float)
    tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
    return {
        'name': name,
        'f1_macro':          round(float(f1_score(yte, yp, average='macro')), 4),
        'precision':         round(float(precision_score(yte, yp)), 4),
        'recall':            round(float(recall_score(yte, yp)), 4),
        'roc_auc':           round(float(roc_auc_score(yte, score)), 4),
        'average_precision': round(float(average_precision_score(yte, score)), 4),
        'confusion_matrix':  {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'train_seconds':     round(time.time() - t0, 1),
    }


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier

    print('[load] train/test CSV...')
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    Xtr_text, ytr = load_text(train), train['label'].to_numpy()
    Xte_text, yte = load_text(test), test['label'].to_numpy()
    print(f'  train={len(train)} test={len(test)}')

    print('[tfidf] vectorisation (1-2-grammes, max_features=50000)...')
    t0 = time.time()
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(Xtr_text)
    Xte = vec.transform(Xte_text)
    print(f'  vocab={len(vec.vocabulary_)} ({time.time()-t0:.0f}s)')

    results = []

    print('[LR] entraînement TF-IDF + Régression Logistique...')
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    r = evaluate('TF-IDF + Régression Logistique', lr, Xtr, ytr, Xte, yte, t0)
    print(f'  F1-macro={r["f1_macro"]:.4f} ({r["train_seconds"]:.0f}s)')
    results.append(r)

    print('[SVM] entraînement TF-IDF + Linear SVM (SGD hinge)...')
    t0 = time.time()
    svm = SGDClassifier(loss='hinge', alpha=1e-5, max_iter=50, class_weight='balanced',
                         random_state=0, n_jobs=-1)
    r = evaluate('TF-IDF + Linear SVM', svm, Xtr, ytr, Xte, yte, t0)
    print(f'  F1-macro={r["f1_macro"]:.4f} ({r["train_seconds"]:.0f}s)')
    results.append(r)

    # Référence Continual-DistilBERT (reports/metrics_summary.json), pour comparaison directe.
    distilbert_ref = None
    metrics_path = os.path.join(REPORTS, 'metrics_summary.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        distilbert_ref = {
            'name': 'Continual-DistilBERT (référence, reports/metrics_summary.json)',
            'f1_macro': round(m['test_f1_macro'], 4),
            'precision': round(m['test_precision'], 4),
            'recall': round(m['test_recall'], 4),
            'roc_auc': round(m['test_roc_auc'], 4),
            'average_precision': round(m['test_average_precision'], 4),
            'confusion_matrix': m['confusion_matrix'],
        }

    out = {
        'meta': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'n_train': len(train), 'n_test': len(test),
            'vectorizer': 'TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2, sublinear_tf=True)',
            'note': 'Même split que Continual-DistilBERT (data/processed/{train,test}.csv). '
                    'Aucune donnée inventée.',
        },
        'baselines': results,
        'distilbert_reference': distilbert_ref,
    }
    with open(os.path.join(REPORTS, 'baselines_statiques.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    L = ['# Baselines statiques — comparaison avec Continual-DistilBERT (H2)\n',
         f"_Généré par `scripts/train_baselines.py` le {time.strftime('%Y-%m-%d')} "
         f"— entraînées sur le même split ({len(train)} train / {len(test)} test) que le "
         "modèle Transformer._\n",
         'Répond à un écart identifié dans `reports/CONFORMITE_PROPOSITION.md` (H2, '
         'Tableau 4.1) : ces deux baselines (absentes du dépôt après l\'incident du '
         '26/08/2026) sont ici réellement ré-entraînées et évaluées sur '
         '`data/processed/test/test.csv`.\n',
         '| Modèle | F1-macro | Précision | Rappel | AUC-ROC | Avg. Precision |',
         '|---|---|---|---|---|---|']
    rows = results + ([distilbert_ref] if distilbert_ref else [])
    for r in rows:
        L.append(f"| {r['name']} | {r['f1_macro']:.4f} | {r['precision']:.4f} | "
                  f"{r['recall']:.4f} | {r['roc_auc']:.4f} | {r['average_precision']:.4f} |")
    if distilbert_ref:
        gain = distilbert_ref['f1_macro'] - max(r['f1_macro'] for r in results)
        L += ['', f"**Écart Continual-DistilBERT vs meilleure baseline classique : "
                  f"{gain:+.4f} point de F1-macro.**"]
    with open(os.path.join(REPORTS, 'baselines_statiques.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"\n[md] {os.path.join(REPORTS, 'baselines_statiques.md')}")
    print('\n==== RÉSUMÉ ====')
    for r in rows:
        print(f"  {r['name']:55s} F1={r['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
