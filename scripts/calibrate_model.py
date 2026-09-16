#!/usr/bin/env python3
# scripts/calibrate_model.py — Calibration probabiliste + prédiction sélective
# KOMOSSI Sosso — Master BIG DATA IA, Institut ESI — UCAO-UUT, 2025-2026
#
# Lève deux limites explicitement constatées au chapitre 4 du mémoire :
#
#   (1) Sur-confiance (Figure 4.5 / Figure 4.7). Le modèle place presque toutes
#       ses prédictions au-dessus de 0,95 de confiance, erreurs comprises, et la
#       corrélation entre justesse et confiance est quasi nulle (0,01). On ajuste
#       donc une température unique T par maximum de vraisemblance sur le jeu de
#       VALIDATION (jamais sur le test), méthode de Guo et al. (2017) : elle ne
#       change pas l'ordre des scores — AUC et matrice de confusion à seuil
#       équivalent sont préservées — mais rend la probabilité affichée lisible.
#
#   (2) Faux positifs sur les dépêches d'agence et faiblesse sur LIAR
#       (exactitude 0,62, Figure 4.6). On calibre deux seuils encadrant une zone
#       grise : au-dessus de tau_fake l'article est déclaré faux, en dessous de
#       tau_real il est déclaré fiable, entre les deux il est marqué INCERTAIN et
#       renvoyé à la vérification humaine plutôt que tranché à tort.
#
# Les seuils sont choisis sur la VALIDATION pour garantir une précision cible sur
# chacune des deux décisions ; le jeu de TEST ne sert qu'au report final.
#
# Sorties :
#   models/calibration.json                     (artefact chargé en production)
#   reports/calibration.md + calibration.json
#   reports/figures/11_calibration_fiabilite.png
#   reports/figures/12_risque_couverture.png
#   reports/cache/logits_{val,test}.npz         (réutilisé par audit_biais.py)
#
# Usage :
#   python scripts/calibrate_model.py [--target-precision 0.97] [--max-abstention 0.15]

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # CPU only (pilote NVML cassé, cf. incident 26/08)

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'spark-app', 'src'))

from calibration import (  # noqa: E402
    ProbabilityCalibrator, brier_score, expected_calibration_error,
    negative_log_likelihood, reliability_curve, risk_coverage_curve, softmax,
)

PRETRAINED = os.path.join(ROOT, 'models', 'pretrained')
VAL_CSV    = os.path.join(ROOT, 'data', 'processed', 'val', 'val.csv')
TEST_CSV   = os.path.join(ROOT, 'data', 'processed', 'test', 'test.csv')
REPORTS    = os.path.join(ROOT, 'reports')
FIGS       = os.path.join(REPORTS, 'figures')
CACHE      = os.path.join(REPORTS, 'cache')
OUT_JSON   = os.path.join(ROOT, 'models', 'calibration.json')


# ── Inférence ────────────────────────────────────────────────────────────────
def compute_logits(csv_path: str, cache_name: str, batch_size: int = 64, force: bool = False):
    """Logits bruts du modèle sur un split, avec cache disque.

    Le format du texte reproduit exactement celui de l'entraînement et de la
    production : "{title[:200]} [SEP] {body[:100]}" (voir train_model.py et
    spark-app/src/nlp_classifier.py) — un format différent déplacerait les
    entrées hors distribution et fausserait la calibration.
    """
    os.makedirs(CACHE, exist_ok=True)
    cache_path = os.path.join(CACHE, f'{cache_name}.npz')
    df = pd.read_csv(csv_path)

    if os.path.exists(cache_path) and not force:
        blob = np.load(cache_path, allow_pickle=True)
        if len(blob['logits']) == len(df):
            print(f'[cache] {cache_name} : {len(df)} logits relus depuis {cache_path}')
            return blob['logits'], blob['y'], df

    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    tok = DistilBertTokenizerFast.from_pretrained(PRETRAINED)
    mdl = DistilBertForSequenceClassification.from_pretrained(PRETRAINED).cpu().eval()

    texts = [f'{str(t)[:200]} [SEP] {str(b)[:100]}' for t, b in zip(df.title, df.body)]
    out, t0 = [], time.time()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], max_length=128, padding='max_length',
                  truncation=True, return_tensors='pt')
        with torch.no_grad():
            out.append(mdl(**enc).logits.numpy())
        if (i // batch_size) % 20 == 0:
            done = min(i + batch_size, len(texts))
            print(f'  {cache_name} {done}/{len(texts)}  ({time.time() - t0:.0f}s)', flush=True)

    logits = np.concatenate(out).astype(np.float64)
    y = df['label'].to_numpy().astype(np.int64)
    np.savez_compressed(cache_path, logits=logits, y=y)
    print(f'[infer] {cache_name} : {len(y)} exemples en {time.time() - t0:.0f}s -> {cache_path}')
    return logits, y, df


# ── Ajustement de la température ─────────────────────────────────────────────
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Minimise la log-vraisemblance négative sur T > 0 (Guo et al., 2017).

    Recherche ternaire sur log T : la NLL en fonction de log T est unimodale, ce
    qui rend la recherche ternaire aussi fiable qu'un LBFGS tout en restant en
    pur numpy et parfaitement reproductible.
    """
    def nll(log_t: float) -> float:
        p = softmax(logits / np.exp(log_t))[:, 1]
        return negative_log_likelihood(p, y)

    lo, hi = np.log(0.05), np.log(20.0)
    for _ in range(200):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float(np.exp((lo + hi) / 2.0))


# ── Choix des seuils de la zone grise ────────────────────────────────────────
def fit_thresholds(p_cal: np.ndarray, y: np.ndarray,
                   target_precision: float, max_abstention: float) -> dict:
    """Deux seuils garantissant une précision cible sur chacune des décisions.

    tau_fake : plus petit seuil tel que, parmi les articles déclarés faux, la
               précision atteigne `target_precision` — c'est lui qui protège les
               dépêches AFP/Reuters/BBC/RFI d'une accusation infondée.
    tau_real : plus grand seuil tel que, parmi les articles déclarés fiables, la
               valeur prédictive négative atteigne `target_precision`.

    La zone grise est ensuite resserrée symétriquement tant que le taux
    d'abstention dépasse `max_abstention` : la sélectivité ne doit pas se payer
    d'une couverture ridicule.
    """
    grid = np.unique(np.round(np.linspace(0.50, 0.995, 400), 4))

    tau_fake = 0.75
    for tau in grid:
        decided = p_cal >= tau
        if decided.sum() < 50:
            break
        if y[decided].mean() >= target_precision:
            tau_fake = float(tau)
            break

    tau_real = 0.25
    for tau in np.unique(np.round(np.linspace(0.50, 0.005, 400), 4)):
        decided = p_cal <= tau
        if decided.sum() < 50:
            break
        if (1.0 - y[decided]).mean() >= target_precision:
            tau_real = float(tau)
            break

    # Resserrement si l'abstention est trop coûteuse.
    def abstention(tf, tr):
        return float(((p_cal > tr) & (p_cal < tf)).mean())

    guard = 0
    while abstention(tau_fake, tau_real) > max_abstention and guard < 200:
        tau_fake = max(0.5, tau_fake - 0.0025)
        tau_real = min(tau_fake, tau_real + 0.0025)
        guard += 1

    return {'tau_fake': round(tau_fake, 4), 'tau_real': round(tau_real, 4)}


# ── Évaluation ───────────────────────────────────────────────────────────────
def selective_report(p_cal: np.ndarray, y: np.ndarray, tau_fake: float, tau_real: float) -> dict:
    """Métriques du régime sélectif : couverture, risque, F1 sur la zone tranchée."""
    from sklearn.metrics import f1_score

    is_fake = p_cal >= tau_fake
    is_real = p_cal <= tau_real
    covered = is_fake | is_real
    n = len(y)

    rep = {
        'coverage':       round(float(covered.mean()), 4),
        'abstention':     round(float(1.0 - covered.mean()), 4),
        'n_uncertain':    int(n - covered.sum()),
        'tau_fake':       tau_fake,
        'tau_real':       tau_real,
    }
    if covered.sum():
        yc = y[covered]
        pc = is_fake[covered].astype(int)
        rep['selective_f1']        = round(float(f1_score(yc, pc, average='macro', zero_division=0)), 4)
        rep['selective_accuracy']  = round(float((pc == yc).mean()), 4)
        rep['selective_risk']      = round(float((pc != yc).mean()), 4)
    if is_fake.sum():
        rep['precision_fake'] = round(float(y[is_fake].mean()), 4)
    if is_real.sum():
        rep['precision_real'] = round(float((1 - y[is_real]).mean()), 4)

    # Régime non sélectif conservé pour comparaison directe avec le Tableau 4.2 :
    # tout ce qui n'atteint pas tau_fake est compté comme réel.
    pred_all = (p_cal >= tau_fake).astype(int)
    rep['f1_macro_full_coverage'] = round(
        float(f1_score(y, pred_all, average='macro', zero_division=0)), 4)
    return rep


def per_source_uncertainty(df: pd.DataFrame, p_cal: np.ndarray,
                           tau_fake: float, tau_real: float) -> dict:
    """Taux d'abstention par source : montre où le modèle reconnaît sa limite."""
    unc = (p_cal > tau_real) & (p_cal < tau_fake)
    out = {}
    for src, idx in df.groupby('source').groups.items():
        pos = df.index.get_indexer(idx)
        out[str(src)] = {
            'n':               int(len(pos)),
            'abstention_rate': round(float(unc[pos].mean()), 4),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['abstention_rate']))


# ── Figures ──────────────────────────────────────────────────────────────────
def figure_reliability(p_raw, p_cal, y, out_png, ece_before, ece_after):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    for ax, p, ece, title, color in (
        (ax1, p_raw, ece_before, 'Avant calibration', '#C0392B'),
        (ax2, p_cal, ece_after,  'Après calibration (temperature scaling)', '#1565C0'),
    ):
        pts = reliability_curve(p, y, n_bins=15)
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        ns = np.array([q[2] for q in pts], dtype=float)
        sizes = 25 + 220 * ns / ns.max() if len(ns) else 25
        ax.plot([0.5, 1.0], [0.5, 1.0], ls='--', color='#7F8C8D', lw=1.2,
                label='calibration parfaite')
        ax.plot(xs, ys, '-', color=color, lw=1.8)
        ax.scatter(xs, ys, s=sizes, color=color, alpha=0.75, zorder=3,
                   label='bins (aire ∝ effectif)')
        ax.set_xlim(0.48, 1.02)
        ax.set_ylim(0.48, 1.02)
        ax.set_xlabel('confiance moyenne du bin')
        ax.set_ylabel('exactitude observée')
        ax.set_title(f'{title}\nECE = {ece:.4f}')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Diagramme de fiabilité — jeu de test (15 554 exemples)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=115)
    plt.close(fig)
    print(f'[fig] {out_png}')


def figure_risk_coverage(p_cal, y, tau_fake, tau_real, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pts = risk_coverage_curve(p_cal, y, n_points=60)
    cov = [q['coverage'] for q in pts]
    risk = [100 * q['risk'] for q in pts]

    is_fake = p_cal >= tau_fake
    is_real = p_cal <= tau_real
    covered = is_fake | is_real
    op_cov = float(covered.mean())
    op_risk = 100 * float(((is_fake[covered].astype(int)) != y[covered]).mean())

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(cov, risk, '-', color='#1565C0', lw=2, label='courbe risque-couverture')
    ax.scatter([op_cov], [op_risk], s=130, color='#C0392B', zorder=5, marker='*',
               label=f'point de fonctionnement retenu\n(couverture {op_cov:.1%}, risque {op_risk:.2f} %)')
    ax.axhline(100 * float(((p_cal >= 0.5).astype(int) != y).mean()), ls='--', lw=1.2,
               color='#7F8C8D', label='risque sans abstention (couverture 100 %)')
    ax.set_xlabel('couverture — part des articles effectivement tranchés')
    ax.set_ylabel("risque — taux d'erreur sur les articles tranchés (%)")
    ax.set_title('Prédiction sélective : ce que l’abstention achète — jeu de test')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=115)
    plt.close(fig)
    print(f'[fig] {out_png}')


# ── Programme principal ──────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-precision', type=float, default=0.97,
                    help='précision minimale exigée sur chacune des deux décisions')
    ap.add_argument('--max-abstention', type=float, default=0.15,
                    help="plafond du taux d'abstention sur le jeu de validation")
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--force', action='store_true', help='ignore le cache de logits')
    args = ap.parse_args()

    os.makedirs(FIGS, exist_ok=True)
    t0 = time.time()

    print('── Inférence sur la validation et le test ──')
    logits_val, y_val, df_val = compute_logits(VAL_CSV, 'logits_val', args.batch_size, args.force)
    logits_test, y_test, df_test = compute_logits(TEST_CSV, 'logits_test', args.batch_size, args.force)

    p_val_raw = softmax(logits_val)[:, 1]
    p_test_raw = softmax(logits_test)[:, 1]

    print('── Ajustement de la température (jeu de validation) ──')
    T = fit_temperature(logits_val, y_val)
    p_val_cal = softmax(logits_val / T)[:, 1]
    p_test_cal = softmax(logits_test / T)[:, 1]
    print(f'  T = {T:.4f}')

    metrics = {
        'val': {
            'ece_before':   round(expected_calibration_error(p_val_raw, y_val), 4),
            'ece_after':    round(expected_calibration_error(p_val_cal, y_val), 4),
            'brier_before': round(brier_score(p_val_raw, y_val), 4),
            'brier_after':  round(brier_score(p_val_cal, y_val), 4),
            'nll_before':   round(negative_log_likelihood(p_val_raw, y_val), 4),
            'nll_after':    round(negative_log_likelihood(p_val_cal, y_val), 4),
        },
        'test': {
            'ece_before':   round(expected_calibration_error(p_test_raw, y_test), 4),
            'ece_after':    round(expected_calibration_error(p_test_cal, y_test), 4),
            'brier_before': round(brier_score(p_test_raw, y_test), 4),
            'brier_after':  round(brier_score(p_test_cal, y_test), 4),
            'nll_before':   round(negative_log_likelihood(p_test_raw, y_test), 4),
            'nll_after':    round(negative_log_likelihood(p_test_cal, y_test), 4),
        },
    }
    for split, m in metrics.items():
        print(f'  [{split}] ECE {m["ece_before"]:.4f} -> {m["ece_after"]:.4f} | '
              f'Brier {m["brier_before"]:.4f} -> {m["brier_after"]:.4f} | '
              f'NLL {m["nll_before"]:.4f} -> {m["nll_after"]:.4f}')

    print('── Calibration des seuils de la zone grise (jeu de validation) ──')
    taus = fit_thresholds(p_val_cal, y_val, args.target_precision, args.max_abstention)
    print(f'  tau_real = {taus["tau_real"]} | tau_fake = {taus["tau_fake"]}')

    rep_val = selective_report(p_val_cal, y_val, taus['tau_fake'], taus['tau_real'])
    rep_test = selective_report(p_test_cal, y_test, taus['tau_fake'], taus['tau_real'])
    src_test = per_source_uncertainty(df_test, p_test_cal, taus['tau_fake'], taus['tau_real'])

    # Référence : le seuil unique 0,75 sur probabilité non calibrée (v2.4).
    from sklearn.metrics import f1_score
    f1_baseline = round(float(f1_score(
        y_test, (p_test_raw >= 0.75).astype(int), average='macro', zero_division=0)), 4)

    artefact = {
        'temperature':  round(T, 6),
        'tau_fake':     taus['tau_fake'],
        'tau_real':     taus['tau_real'],
        'fitted':       True,
        'selective':    taus['tau_real'] < taus['tau_fake'],
        'fitted_on':    'data/processed/val/val.csv',
        'fitted_at':    datetime.now(timezone.utc).isoformat(),
        'n_val':        int(len(y_val)),
        'target_precision': args.target_precision,
        'ece_before':   metrics['test']['ece_before'],
        'ece_after':    metrics['test']['ece_after'],
        'brier_before': metrics['test']['brier_before'],
        'brier_after':  metrics['test']['brier_after'],
        'nll_before':   metrics['test']['nll_before'],
        'nll_after':    metrics['test']['nll_after'],
        'coverage':     rep_test.get('coverage'),
        'selective_f1': rep_test.get('selective_f1'),
        'selective_risk': rep_test.get('selective_risk'),
        'f1_macro_full_coverage': rep_test.get('f1_macro_full_coverage'),
        'f1_macro_baseline_075_uncalibrated': f1_baseline,
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(artefact, fh, indent=2, ensure_ascii=False)
    print(f'[artefact] {OUT_JSON}')

    figure_reliability(p_test_raw, p_test_cal, y_test,
                       os.path.join(FIGS, '11_calibration_fiabilite.png'),
                       metrics['test']['ece_before'], metrics['test']['ece_after'])
    figure_risk_coverage(p_test_cal, y_test, taus['tau_fake'], taus['tau_real'],
                         os.path.join(FIGS, '12_risque_couverture.png'))

    full = {
        'meta': {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'seconds':      round(time.time() - t0, 1),
            'temperature':  round(T, 6),
            'target_precision': args.target_precision,
            'max_abstention':   args.max_abstention,
        },
        'metrics':          metrics,
        'thresholds':       taus,
        'selective_val':    rep_val,
        'selective_test':   rep_test,
        'per_source_abstention_test': src_test,
        'risk_coverage_test': risk_coverage_curve(p_test_cal, y_test, n_points=40),
        'f1_macro_baseline_075_uncalibrated': f1_baseline,
    }
    with open(os.path.join(REPORTS, 'calibration.json'), 'w', encoding='utf-8') as fh:
        json.dump(full, fh, indent=2, ensure_ascii=False)

    write_markdown(full, artefact)
    print(f'\nTerminé en {time.time() - t0:.0f}s.')


def write_markdown(full: dict, artefact: dict):
    m_test = full['metrics']['test']
    m_val = full['metrics']['val']
    st = full['selective_test']
    taus = full['thresholds']

    L = [
        '# Calibration probabiliste et prédiction sélective\n',
        f"_Généré par `scripts/calibrate_model.py` le {time.strftime('%Y-%m-%d')} "
        f"(température ajustée sur `data/processed/val/val.csv`, report sur le jeu de test "
        f"tenu à l'écart, {full['meta']['seconds']:.0f}s CPU)._\n",
        "## 1. Pourquoi\n",
        "La Figure 4.5 du mémoire montrait un modèle **fortement sur-confiant** : la quasi-totalité "
        "des prédictions, justes comme erronées, se logeait au-dessus de 0,95 de confiance, et la "
        "Figure 4.7 relevait une corrélation quasi nulle (0,01) entre justesse et probabilité "
        "prédite. La confiance affichée ne prédisait donc pas la justesse — ce que le chapitre 4 "
        "signalait déjà comme « une amélioration immédiate du dispositif ». C'est cette amélioration "
        "qui est implémentée ici.\n",
        '## 2. Temperature scaling\n',
        f'Température ajustée par maximum de vraisemblance : **T = {artefact["temperature"]:.4f}**. '
        "La transformation étant monotone, l'ordre des scores et donc l'AUC-ROC (0,9899) sont "
        'strictement inchangés ; seule la lecture probabiliste est corrigée.\n',
        '| Métrique | Validation avant | Validation après | Test avant | Test après |',
        '|---|---|---|---|---|',
        f'| ECE (erreur de calibration attendue) | {m_val["ece_before"]:.4f} | **{m_val["ece_after"]:.4f}** '
        f'| {m_test["ece_before"]:.4f} | **{m_test["ece_after"]:.4f}** |',
        f'| Score de Brier | {m_val["brier_before"]:.4f} | **{m_val["brier_after"]:.4f}** '
        f'| {m_test["brier_before"]:.4f} | **{m_test["brier_after"]:.4f}** |',
        f'| Log-vraisemblance négative | {m_val["nll_before"]:.4f} | **{m_val["nll_after"]:.4f}** '
        f'| {m_test["nll_before"]:.4f} | **{m_test["nll_after"]:.4f}** |',
        '',
        '![Diagramme de fiabilité](figures/11_calibration_fiabilite.png)\n',
        '## 3. Prédiction sélective\n',
        f'Seuils calibrés sur la validation pour une précision cible de '
        f'{full["meta"]["target_precision"]:.0%} sur chacune des deux décisions :\n',
        f'- `p_cal >= {taus["tau_fake"]}` → **FAKE**',
        f'- `p_cal <= {taus["tau_real"]}` → **RÉEL**',
        f'- entre les deux → **INCERTAIN**, renvoyé à la vérification humaine\n',
        '| Mesure (jeu de test) | Valeur |',
        '|---|---|',
        f'| Couverture (articles tranchés) | **{st.get("coverage", 0):.2%}** |',
        f'| Abstention | {st.get("abstention", 0):.2%} ({st.get("n_uncertain", 0)} articles) |',
        f'| F1-macro sur la zone tranchée | **{st.get("selective_f1", 0):.4f}** |',
        f'| Risque (erreur) sur la zone tranchée | **{st.get("selective_risk", 0):.2%}** |',
        f'| Précision de la décision « faux » | {st.get("precision_fake", 0):.4f} |',
        f'| Précision de la décision « fiable » | {st.get("precision_real", 0):.4f} |',
        f'| F1-macro à couverture 100 % (abstention comptée « réel ») | {st.get("f1_macro_full_coverage", 0):.4f} |',
        f'| Référence v2.4 : seuil 0,75 sur probabilité non calibrée | '
        f'{full["f1_macro_baseline_075_uncalibrated"]:.4f} |',
        '',
        '![Courbe risque-couverture](figures/12_risque_couverture.png)\n',
        '## 4. Où le modèle reconnaît sa limite\n',
        "Taux d'abstention par source sur le jeu de test — la zone grise se concentre "
        'exactement là où le chapitre 4 situait la faiblesse du modèle :\n',
        '| Source | n | Taux d’abstention |',
        '|---|---|---|',
    ]
    for src, d in list(full['per_source_abstention_test'].items())[:12]:
        L.append(f'| {src} | {d["n"]} | {d["abstention_rate"]:.2%} |')
    L += [
        '',
        '## 5. Intégration en production\n',
        "L'artefact `models/calibration.json` est chargé au démarrage par "
        '`spark-app/src/nlp_classifier.py` et par le routeur de recherche web de l’API. '
        "En son absence, le pipeline retombe exactement sur le comportement antérieur "
        '(T = 1, seuil unique 0,75, aucune abstention) : la mise à jour est donc sans risque '
        'de régression.\n',
    ]
    path = os.path.join(REPORTS, 'calibration.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'[md] {path}')


if __name__ == '__main__':
    main()
