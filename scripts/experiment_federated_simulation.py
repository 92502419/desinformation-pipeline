#!/usr/bin/env python3
# scripts/experiment_federated_simulation.py
# Preuve de concept pour la Recommandation 3 du memoire (Chapitre 5, §5.5) :
# "Federated Learning pour la collaboration inter-organisations".
#
# Portee assumee et honnete : federer reellement Continual-DistilBERT (135M
# parametres) entre plusieurs organisations demanderait une orchestration
# reseau reelle (gRPC/Flower), plusieurs machines, et des heures de calcul —
# hors de portee du materiel de developpement (CPU seul, incident GPU du
# 26/08/2026). Ce script demontre neanmoins le MECANISME de federated
# averaging (FedAvg, McMahan et al. 2017) sur un modele plus leger
# (HashingVectorizer + SGDClassifier logistique), avec de VRAIES donnees
# partitionnees par organisation simulee (source du corpus). Aucune
# communication de donnees brutes entre "organisations" : seuls les poids du
# modele local (coef_/intercept_) sont moyennes, exactement comme le ferait
# une vraie federation.
#
# 4 organisations simulees, partitionnees par source (aucun partage de texte
# entre elles) :
#   - org_welfake      : articles WELFake (registre journalistique anglophone)
#   - org_fakenewsnet  : articles FakeNewsNet (index + texte exploitable)
#   - org_liar         : articles LIAR (declarations politiques courtes)
#   - org_afrique      : sous-corpus africain (MasakhaNEWS + Google News RSS)
#
# Compare 3 regimes sur le MEME jeu de test (data/processed/test/test.csv) :
#   1. SOLO       : chaque organisation entraine sur ses seules donnees, sans
#                   collaboration -> generalise mal hors de son propre registre.
#   2. FEDERE     : FedAvg sur R rounds, moyenne ponderee par taille locale des
#                   coef_/intercept_ apres chaque round de mise a jour locale.
#                   Aucune donnee brute ne quitte une organisation.
#   3. CENTRALISE : meme modele entraine sur les donnees poolees (upper bound
#                   theorique, non atteignable dans un vrai scenario federe car
#                   il suppose un partage complet des donnees).
#
# Sortie : reports/experience_federated_simulation.{json,md}
#          reports/figures/13_federated_simulation.png
#
# Usage : python scripts/experiment_federated_simulation.py [--rounds 15]

import os, sys, json, time, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, 'data/processed/train/train.csv')
TEST_CSV  = os.path.join(ROOT, 'data/processed/test/test.csv')
REPORTS   = os.path.join(ROOT, 'reports')
FIGS      = os.path.join(REPORTS, 'figures')

N_FEATURES = 2 ** 18   # HashingVectorizer : aucun vocabulaire a partager entre organisations

ORG_MAP = {
    'welfake':     'org_welfake',
    'fakenewsnet': 'org_fakenewsnet',
    'liar':        'org_liar',
}


def assign_org(source: str) -> str:
    if source in ORG_MAP:
        return ORG_MAP[source]
    if source.startswith('masakhanews') or source == 'google_news_africa_rss':
        return 'org_afrique'
    return 'org_autre'


def load_text(df):
    title = df['title'].fillna('').astype(str)
    body = df['body'].fillna('').astype(str)
    return (title + ' ' + body).str.slice(0, 2000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=15)
    ap.add_argument('--local_epochs', type=int, default=1)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

    os.makedirs(FIGS, exist_ok=True)
    t0 = time.time()

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train['org'] = train['source'].astype(str).apply(assign_org)
    train = train[train['org'] != 'org_autre']
    orgs = sorted(train['org'].unique())
    sizes = train['org'].value_counts().to_dict()
    print(f'[orgs] {len(orgs)} organisations simulees : '
          + ', '.join(f'{o}={sizes[o]}' for o in orgs), flush=True)

    vec = HashingVectorizer(n_features=N_FEATURES, alternate_sign=False, ngram_range=(1, 2))
    Xte = vec.transform(load_text(test))
    yte = test['label'].to_numpy()
    Xtr_all = vec.transform(load_text(train))
    ytr_all = train['label'].to_numpy()

    org_data = {}
    for o in orgs:
        mask = (train['org'] == o).to_numpy()
        org_data[o] = (vec.transform(load_text(train[mask])), train.loc[mask, 'label'].to_numpy())

    def new_model():
        return SGDClassifier(loss='log_loss', alpha=1e-5, random_state=0,
                             learning_rate='optimal', class_weight='balanced')

    def new_model_federated():
        # class_weight='balanced' n'est pas supporte par partial_fit (utilise pour
        # le regime FEDERE) -> pondere manuellement via sample_weight ci-dessous.
        return SGDClassifier(loss='log_loss', alpha=1e-5, random_state=0, learning_rate='optimal')

    def local_sample_weight(y):
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        if len(classes) < 2:
            return np.ones_like(y, dtype=float)
        cw = compute_class_weight('balanced', classes=classes, y=y)
        cw_map = dict(zip(classes, cw))
        return np.array([cw_map[v] for v in y], dtype=float)

    def eval_model(m, name):
        yp = m.predict(Xte)
        tn, fp, fn, tp = confusion_matrix(yte, yp).ravel()
        return {
            'name': name,
            'f1_macro':  round(float(f1_score(yte, yp, average='macro')), 4),
            'precision': round(float(precision_score(yte, yp)), 4),
            'recall':    round(float(recall_score(yte, yp)), 4),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        }

    # ── 1. SOLO : chaque organisation entraine seule, evaluee sur le meme test ──
    print('[solo] entrainement independant par organisation...', flush=True)
    solo_results = []
    for o in orgs:
        Xo, yo = org_data[o]
        if len(np.unique(yo)) < 2:
            # Cas reel et documente (README/memoire) : le sous-corpus africain ne
            # contient aucun exemple fake (Africa Check inaccessible, 403 Cloudflare).
            # Un tel participant ne peut tout simplement PAS entrainer de detecteur
            # seul -> illustration concrete de l'interet de la federation.
            r = {'name': f'SOLO — {o}', 'f1_macro': None, 'precision': None, 'recall': None,
                 'confusion_matrix': None,
                 'note': f'corpus mono-classe (label={int(yo[0])} uniquement) — entrainement solo impossible'}
            solo_results.append(r)
            print(f"  {o:20s} n={len(yo):6d}  F1=N/A (mono-classe, label={int(yo[0])})", flush=True)
            continue
        m = new_model()
        m.fit(Xo, yo)
        r = eval_model(m, f'SOLO — {o}')
        solo_results.append(r)
        print(f"  {o:20s} n={len(yo):6d}  F1={r['f1_macro']:.4f}", flush=True)

    # ── 2. FEDERE : FedAvg — R rounds de mise a jour locale + moyenne ponderee ──
    print(f'[federe] FedAvg sur {args.rounds} rounds...', flush=True)
    global_model = new_model_federated()
    # Initialisation : un partial_fit factice pour allouer coef_/intercept_
    global_model.partial_fit(Xtr_all[:2], ytr_all[:2], classes=np.array([0, 1]))
    global_model.coef_[:] = 0.0
    global_model.intercept_[:] = 0.0

    fed_curve = []
    total_n = sum(sizes[o] for o in orgs)
    for rnd in range(1, args.rounds + 1):
        local_coefs, local_intercepts, weights = [], [], []
        for o in orgs:
            Xo, yo = org_data[o]
            sw = local_sample_weight(yo)
            local = new_model_federated()
            local.partial_fit(Xo[:2], yo[:2], classes=np.array([0, 1]))
            local.coef_ = global_model.coef_.copy()
            local.intercept_ = global_model.intercept_.copy()
            for _ in range(args.local_epochs):
                local.partial_fit(Xo, yo, sample_weight=sw)
            local_coefs.append(local.coef_)
            local_intercepts.append(local.intercept_)
            weights.append(sizes[o] / total_n)
        global_model.coef_ = sum(w * c for w, c in zip(weights, local_coefs))
        global_model.intercept_ = sum(w * c for w, c in zip(weights, local_intercepts))
        r = eval_model(global_model, f'FEDERE — round {rnd}')
        fed_curve.append({'round': rnd, 'f1_macro': r['f1_macro']})
        if rnd % 3 == 0 or rnd == 1:
            print(f"  round {rnd:2d}/{args.rounds}  F1(global)={r['f1_macro']:.4f}", flush=True)
    federated_result = eval_model(global_model, 'FEDERE (FedAvg, aucune donnee brute partagee)')

    # ── 3. CENTRALISE : upper bound theorique (donnees poolees) ──────────────
    print('[centralise] entrainement sur donnees poolees (upper bound)...', flush=True)
    centralized = new_model()
    centralized.fit(Xtr_all, ytr_all)
    centralized_result = eval_model(centralized, 'CENTRALISE (upper bound, donnees poolees)')

    # ── Rapport ────────────────────────────────────────────────────────────
    out = {
        'meta': {
            'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'rounds': args.rounds, 'local_epochs': args.local_epochs,
            'n_features_hashing': N_FEATURES,
            'organizations': {o: int(sizes[o]) for o in orgs},
            'n_test': len(test),
            'seconds': round(time.time() - t0, 1),
            'note': ('Preuve de concept FedAvg (HashingVectorizer + SGDClassifier logistique). '
                     'Federer reellement Continual-DistilBERT entre organisations est hors de '
                     "portee du materiel de developpement (CPU seul) — voir memoire, Chapitre 5, "
                     'Recommandation 3.'),
        },
        'solo': solo_results,
        'federated': federated_result,
        'federated_curve': fed_curve,
        'centralized': centralized_result,
    }
    with open(os.path.join(REPORTS, 'experience_federated_simulation.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # figure
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fc = pd.DataFrame(fed_curve)
    ax1.plot(fc['round'], fc['f1_macro'], 'o-', color='#1565C0', label='FEDERE (FedAvg)')
    ax1.axhline(centralized_result['f1_macro'], color='#27AE60', ls='--',
                label=f"CENTRALISE ({centralized_result['f1_macro']:.4f})")
    evaluable_solo = [r for r in solo_results if r['f1_macro'] is not None]
    best_solo = max(r['f1_macro'] for r in evaluable_solo)
    ax1.axhline(best_solo, color='#C0392B', ls=':', label=f'Meilleur SOLO ({best_solo:.4f})')
    ax1.set_xlabel('Round FedAvg'); ax1.set_ylabel('F1-macro (jeu de test commun)')
    ax1.set_title('Convergence du modele federe'); ax1.legend(fontsize=8)

    names = [o.replace('org_', '') for o in orgs] + ['FEDERE', 'CENTRALISE']
    vals = [(r['f1_macro'] if r['f1_macro'] is not None else 0.0) for r in solo_results] + \
           [federated_result['f1_macro'], centralized_result['f1_macro']]
    colors = ['#C0392B'] * len(orgs) + ['#1565C0', '#27AE60']
    ax2.bar(names, vals, color=colors)
    ax2.set_ylabel('F1-macro'); ax2.set_ylim(0, 1.02)
    ax2.set_title('SOLO vs FEDERE vs CENTRALISE (jeu de test commun)')
    ax2.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, '13_federated_simulation.png'), dpi=115)
    print(f"[fig] {os.path.join(FIGS, '13_federated_simulation.png')}", flush=True)

    L = ['# Preuve de concept — Federated Learning inter-organisations (Recommandation 3)\n',
         f"_Genere par `scripts/experiment_federated_simulation.py` le {time.strftime('%Y-%m-%d')} "
         f"({args.rounds} rounds FedAvg, {out['meta']['seconds']:.0f}s CPU)._\n",
         'Repond a la Recommandation 3 du memoire (Chapitre 5, §5.5) par une preuve de '
         'concept reelle et executable, a defaut de federer Continual-DistilBERT lui-meme '
         '(hors de portee CPU). 4 organisations simulees par partition du corpus par '
         'source ; **aucune donnee brute ne quitte une organisation** — seuls les poids du '
         "modele local sont moyennes (FedAvg, McMahan et al. 2017), exactement comme le "
         'ferait une federation reelle.\n',
         '| Regime | n | F1-macro (jeu de test commun) |',
         '|---|---|---|']
    for r, o in zip(solo_results, orgs):
        f1_txt = f"{r['f1_macro']:.4f}" if r['f1_macro'] is not None else f"N/A — {r.get('note', 'non évaluable')}"
        L.append(f"| SOLO — {o} | {sizes[o]} | {f1_txt} |")
    L.append(f"| **FEDERE (FedAvg, {args.rounds} rounds)** | {total_n} (jamais poole) | "
             f"**{federated_result['f1_macro']:.4f}** |")
    L.append(f"| CENTRALISE (upper bound theorique) | {total_n} (poole) | "
             f"{centralized_result['f1_macro']:.4f} |")
    gain_vs_best_solo = federated_result['f1_macro'] - best_solo
    gap_vs_centralized = centralized_result['f1_macro'] - federated_result['f1_macro']
    L += ['',
          f'- **Gain du régime fédéré vs meilleur régime solo** : {gain_vs_best_solo:+.4f} '
          'point de F1-macro, sans qu\'aucune organisation ne partage ses données brutes '
          '— et surtout, `org_afrique` (mono-classe) ne peut tout simplement pas entraîner '
          'de détecteur seule : la fédération est la seule façon pour elle de bénéficier '
          "d'un modèle fonctionnel sans exposer ses données.",
          f"- **Écart au majorant centralisé** : {gap_vs_centralized:+.4f} point. Cet écart "
          "important est cohérent avec la littérature FedAvg (McMahan et al. 2017) : le "
          'partitionnement par source ici est fortement **non-IID** (registres textuels très '
          'différents — dépêches WELFake, déclarations politiques LIAR, presse africaine '
          "100 % réelle) et une simple moyenne pondérée des poids locaux converge mal dans "
          'ce régime, phénomène documenté et toujours actif en recherche (FedProx, SCAFFOLD, '
          "et d'autres variantes visent précisément à corriger cette limite de FedAvg vanilla).",
          '', f"![Simulation FedAvg](figures/13_federated_simulation.png)\n",
          '## Limites de cette preuve de concept\n',
          '- Modèle plus léger (HashingVectorizer + régression logistique SGD) que '
          'Continual-DistilBERT, choisi pour rester exécutable en quelques minutes sur '
          'CPU sans infrastructure réseau réelle.',
          "- Simulation en un seul processus (4 \"organisations\" = 4 partitions en mémoire) : "
          'aucune vraie communication réseau, aucun chiffrement, aucune gestion de la '
          "latence/panne d'un participant — tout cela relève d'une implémentation "
          'production (ex. Flower, TensorFlow Federated).',
          '- Le partitionnement par source est un simplificateur pédagogique ; une vraie '
          'fédération inter-organisations poserait aussi des questions de format de '
          'données et de gouvernance non traitées ici.']
    with open(os.path.join(REPORTS, 'experience_federated_simulation.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"[md] {os.path.join(REPORTS, 'experience_federated_simulation.md')}")

    print('\n==== RESUME ====')
    for r in solo_results:
        f1_txt = f"{r['f1_macro']:.4f}" if r['f1_macro'] is not None else 'N/A'
        print(f"  {r['name']:40s} F1={f1_txt}")
    print(f"  {'FEDERE (FedAvg)':40s} F1={federated_result['f1_macro']:.4f}")
    print(f"  {'CENTRALISE (upper bound)':40s} F1={centralized_result['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
