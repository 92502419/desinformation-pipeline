# spark-app/src/calibration.py — Calibration probabiliste + prédiction sélective
# KOMOSSI Sosso — Master BIG DATA IA, Institut ESI — UCAO-UUT, 2025-2026
#
# v3.0 — Répond à deux limites identifiées au chapitre 4 du mémoire :
#   1. Sur-confiance du modèle (Figure 4.5 : quasi toutes les prédictions, justes
#      comme erronées, se logent au-dessus de 0,95 de confiance ; corrélation
#      justesse/confiance ≈ 0,01, Figure 4.7). Correctif : temperature scaling,
#      une recalibration à un seul paramètre T ajusté par maximum de vraisemblance
#      sur le jeu de validation (Guo et al., 2017). T > 1 aplatit les probabilités,
#      T < 1 les durcit ; l'ordre des scores — donc l'AUC — est strictement
#      préservé, seule la lecture probabiliste change.
#   2. Faux positifs sur les dépêches d'agence et faiblesse sur LIAR. Correctif :
#      prédiction sélective — un article dont la probabilité calibrée tombe dans
#      la zone grise [tau_real, tau_fake] n'est plus tranché mais marqué INCERTAIN
#      et renvoyé à la vérification humaine, au lieu d'être accusé à tort.
#
# Ce module ne dépend que de numpy : il est chargé aussi bien par le classifieur
# Spark que par l'API FastAPI (inférence de la recherche web).
#
# ══════════════════════════════════════════════════════════════════════════════
# FICHIER PARTAGÉ — copie strictement identique dans api/src/calibration.py.
# tests/test_shared_modules.py échoue si les deux copies divergent.
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import json
import math
import os
from typing import Sequence

import numpy as np

# Emplacement de l'artefact produit par scripts/calibrate_model.py.
CALIBRATION_PATH = os.getenv('MODEL_CALIBRATION_PATH', '/app/models/calibration.json')

# Repli sûr : température neutre et seuils identiques au comportement v2.4
# (P(fake) >= 0,75 -> fake, sinon réel, aucune abstention). Si l'artefact de
# calibration est absent, le pipeline se comporte donc exactement comme avant.
DEFAULT_CALIBRATION = {
    'temperature':      1.0,
    'tau_fake':         0.75,
    'tau_real':         0.75,
    'fitted':           False,
    'selective':        False,
}

VERDICT_FAKE      = 'fake'
VERDICT_REAL      = 'real'
VERDICT_UNCERTAIN = 'uncertain'


def softmax(logits) -> np.ndarray:
    """Softmax numériquement stable sur le dernier axe."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def binary_entropy(p: float) -> float:
    """Entropie de Shannon (base 2) d'une décision binaire, normalisée dans [0, 1].

    Vaut 0 lorsque le modèle est certain (p = 0 ou p = 1) et 1 lorsqu'il est
    totalement indécis (p = 0,5). C'est le signal d'incertitude alimentant le
    canal « uncertainty » du moniteur de dérive multi-signaux.
    """
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def expected_calibration_error(p_fake, y_true, n_bins: int = 15) -> float:
    """ECE : écart moyen, pondéré par effectif, entre confiance et exactitude.

    La confiance d'une prédiction binaire est max(p, 1-p) ; l'exactitude est la
    fraction de prédictions correctes dans le bin. Un modèle parfaitement
    calibré a un ECE nul.
    """
    p_fake = np.asarray(p_fake, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = (p_fake >= 0.5).astype(np.int64)
    conf = np.maximum(p_fake, 1.0 - p_fake)
    correct = (pred == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_fake)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(p_fake, y_true, n_bins: int = 15):
    """Points (confiance moyenne, exactitude moyenne, effectif) du diagramme de fiabilité."""
    p_fake = np.asarray(p_fake, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = (p_fake >= 0.5).astype(np.int64)
    conf = np.maximum(p_fake, 1.0 - p_fake)
    correct = (pred == y_true).astype(np.float64)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        pts.append((float(conf[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return pts


def brier_score(p_fake, y_true) -> float:
    """Score de Brier : erreur quadratique moyenne sur la probabilité de la classe fake."""
    p_fake = np.asarray(p_fake, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(np.mean((p_fake - y_true) ** 2))


def negative_log_likelihood(p_fake, y_true) -> float:
    """Log-vraisemblance négative moyenne (perte d'entropie croisée)."""
    p = np.clip(np.asarray(p_fake, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class ProbabilityCalibrator:
    """Applique la température apprise et arbitre les trois verdicts possibles.

    Le classifieur reste binaire : `label` vaut toujours 0 ou 1 pour préserver la
    compatibilité de tout l'existant (MongoDB, Elasticsearch, Grafana, API). Le
    champ `verdict` s'y ajoute et porte l'information de sélectivité :
    fake / real / uncertain.
    """

    def __init__(self, params: dict | None = None):
        cfg = dict(DEFAULT_CALIBRATION)
        if params:
            cfg.update({k: v for k, v in params.items() if v is not None})
        self.temperature = max(float(cfg.get('temperature', 1.0)), 1e-3)
        self.tau_fake = float(cfg.get('tau_fake', 0.75))
        self.tau_real = float(cfg.get('tau_real', self.tau_fake))
        # tau_real ne peut pas dépasser tau_fake : sinon la zone grise est vide
        # et le verdict devient ambigu.
        self.tau_real = min(self.tau_real, self.tau_fake)
        self.fitted = bool(cfg.get('fitted', False))
        self.selective = bool(cfg.get('selective', False)) and self.tau_real < self.tau_fake
        self.meta = cfg

    # ── Chargement ───────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str | None = None) -> 'ProbabilityCalibrator':
        """Charge models/calibration.json ; retombe silencieusement sur le
        comportement v2.4 si le fichier est absent ou illisible."""
        path = path or CALIBRATION_PATH
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return cls(json.load(fh))
        except Exception:
            return cls(None)

    # ── Application ──────────────────────────────────────────────────────────
    def apply_logits(self, logits) -> float:
        """Logits bruts (2,) -> probabilité calibrée de la classe fake."""
        z = np.asarray(logits, dtype=np.float64) / self.temperature
        return float(softmax(z)[..., 1])

    def apply_prob(self, p_fake: float) -> float:
        """Recalibre une probabilité déjà passée par softmax.

        Utile lorsque seule la probabilité est disponible (résultats archivés,
        expériences hors ligne) : on reconstruit le logit relatif par la
        fonction logit, on le divise par T, puis on repasse par la sigmoïde.
        """
        p = float(min(max(p_fake, 1e-12), 1.0 - 1e-12))
        logit = math.log(p / (1.0 - p)) / self.temperature
        return float(1.0 / (1.0 + math.exp(-logit)))

    def apply_prob_array(self, p_fake) -> np.ndarray:
        """Version vectorisée de apply_prob."""
        p = np.clip(np.asarray(p_fake, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        logit = np.log(p / (1.0 - p)) / self.temperature
        return 1.0 / (1.0 + np.exp(-logit))

    # ── Décision ─────────────────────────────────────────────────────────────
    def verdict(self, p_cal: float) -> str:
        """fake / real / uncertain à partir de la probabilité calibrée."""
        if p_cal >= self.tau_fake:
            return VERDICT_FAKE
        if self.selective and p_cal > self.tau_real:
            return VERDICT_UNCERTAIN
        return VERDICT_REAL

    def label(self, p_cal: float) -> int:
        """Étiquette binaire rétro-compatible : 1 = fake, 0 = réel.

        Un article incertain reste étiqueté 0 (non accusé) : le principe retenu
        au chapitre 5 est qu'une accusation infondée coûte plus cher qu'une
        détection manquée.
        """
        return 1 if p_cal >= self.tau_fake else 0

    def decide(self, logits=None, p_fake: float | None = None) -> dict:
        """Point d'entrée unique de la décision.

        Fournir soit les logits bruts (chemin normal, calibration exacte), soit
        une probabilité déjà softmaxée (chemin de rattrapage).
        """
        if logits is not None:
            p_raw = float(softmax(np.asarray(logits, dtype=np.float64))[..., 1])
            p_cal = self.apply_logits(logits)
        elif p_fake is not None:
            p_raw = float(p_fake)
            p_cal = self.apply_prob(p_fake)
        else:
            raise ValueError('decide() exige logits ou p_fake')
        verdict = self.verdict(p_cal)
        label = self.label(p_cal)
        return {
            'label':       label,
            'verdict':     verdict,
            'p_fake':      round(p_cal, 6),
            'p_fake_raw':  round(p_raw, 6),
            'confidence':  round(max(p_cal, 1.0 - p_cal), 6),
            'entropy':     round(binary_entropy(p_cal), 6),
            'calibrated':  self.fitted,
        }

    def describe(self) -> dict:
        """Résumé exposé par l'API (/api/v1/model/calibration) et Streamlit."""
        return {
            'temperature':  round(self.temperature, 4),
            'tau_fake':     round(self.tau_fake, 4),
            'tau_real':     round(self.tau_real, 4),
            'fitted':       self.fitted,
            'selective':    self.selective,
            'ece_before':   self.meta.get('ece_before'),
            'ece_after':    self.meta.get('ece_after'),
            'brier_before': self.meta.get('brier_before'),
            'brier_after':  self.meta.get('brier_after'),
            'nll_before':   self.meta.get('nll_before'),
            'nll_after':    self.meta.get('nll_after'),
            'coverage':     self.meta.get('coverage'),
            'selective_f1': self.meta.get('selective_f1'),
            'fitted_on':    self.meta.get('fitted_on'),
            'fitted_at':    self.meta.get('fitted_at'),
        }


def risk_coverage_curve(p_cal: Sequence[float], y_true: Sequence[int], n_points: int = 40):
    """Courbe risque-couverture de la prédiction sélective.

    Pour une zone grise symétrique de demi-largeur croissante autour de 0,5, on
    reporte la couverture (part d'articles effectivement tranchés) et le risque
    (taux d'erreur sur ces seuls articles). Elle quantifie ce que l'abstention
    achète : combien d'erreurs sont évitées pour combien d'articles renvoyés à
    la vérification humaine.
    """
    p = np.asarray(p_cal, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(np.int64)
    err = (pred != y).astype(np.float64)

    pts = []
    for q in np.linspace(0.0, 0.9, n_points):
        cut = float(np.quantile(conf, q))
        keep = conf >= cut
        if keep.sum() == 0:
            continue
        pts.append({
            'confidence_cut': round(cut, 4),
            'coverage':       round(float(keep.mean()), 4),
            'risk':           round(float(err[keep].mean()), 4),
            'n_kept':         int(keep.sum()),
        })
    return pts
