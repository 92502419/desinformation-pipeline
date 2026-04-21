# spark-app/src/drift_monitor.py — Tri-Détecteur de Concept Drift
# KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT 2025-2026
#
# Détecteurs River 0.21.2 disponibles : ADWIN, KSWIN, PageHinkley
# Note : EDDM n'est pas dans River 0.21.2 → remplacé par PageHinkley
#   - ADWIN      (poids 0.45) : dérive abrupte — fenêtre adaptative
#   - KSWIN      (poids 0.35) : dérive statistique — test de Kolmogorov-Smirnov
#   - PageHinkley(poids 0.20) : dérive graduelle — somme cumulée avec seuil

from river import drift
from typing import Dict
from datetime import datetime, timezone
import os, math


ADWIN_DELTA    = float(os.getenv('ADWIN_DELTA', 0.002))
KSWIN_WINDOW   = int(os.getenv('KSWIN_WINDOW_SIZE', 100))
KSWIN_ALPHA    = float(os.getenv('KSWIN_ALPHA', 0.005))
PH_DELTA       = float(os.getenv('PH_DELTA', 0.005))        # sensibilité PageHinkley
PH_THRESHOLD   = float(os.getenv('PH_THRESHOLD', 50.0))     # seuil PageHinkley
DRIFT_THRESH   = float(os.getenv('DRIFT_COMPOSITE_THRESHOLD', 0.5))
CONFIRM_THRESH = float(os.getenv('DRIFT_CONFIRMED_THRESHOLD', 0.8))
LR_BASE        = float(os.getenv('ONLINE_LR_BASE', 1e-5))
LR_DRIFT       = float(os.getenv('ONLINE_LR_DRIFT', 5e-5))


class DynamicDriftMonitor:
    """
    Tri-Détecteur de Concept Drift (River 0.21.2) :
    - ADWIN        (poids 0.45) : surveillance de la moyenne — dérive abrupte
    - KSWIN        (poids 0.35) : test statistique KS — changement de distribution
    - PageHinkley  (poids 0.20) : somme cumulée — dérive graduelle
    """
    WEIGHTS = {'ADWIN': 0.45, 'KSWIN': 0.35, 'PageHinkley': 0.20}

    def __init__(self):
        self.adwin        = drift.ADWIN(delta=ADWIN_DELTA)
        self.kswin        = drift.KSWIN(
            window_size=KSWIN_WINDOW,
            stat_size=KSWIN_WINDOW // 3,
            alpha=KSWIN_ALPHA,
        )
        self.page_hinkley = drift.PageHinkley(
            min_instances=30,
            delta=PH_DELTA,
            threshold=PH_THRESHOLD,
            alpha=1 - 0.0001,
        )
        self.composite_score       = 0.0
        self.drift_active          = False
        self.drift_confirmed       = False
        self.messages_total        = 0
        self.messages_since_drift  = 0
        self.drift_events          = []
        self.confidence_history    = []
        print(
            f'[DRIFT] Monitor initialisé | '
            f'ADWIN δ={ADWIN_DELTA} | KSWIN α={KSWIN_ALPHA} | '
            f'PageHinkley δ={PH_DELTA} λ={PH_THRESHOLD}'
        )

    def update(self, confidence: float, error_bit: int = 0) -> Dict:
        # ── Validation et nettoyage des entrées ──────────────────────
        if confidence is None or (isinstance(confidence, float) and
                                  (math.isnan(confidence) or math.isinf(confidence))):
            confidence = 0.5
        confidence = float(max(0.0, min(1.0, confidence)))

        self.messages_total += 1
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 1000:
            self.confidence_history.pop(0)

        # ── Mise à jour des 3 détecteurs ─────────────────────────────
        self.adwin.update(confidence)
        self.kswin.update(confidence)
        self.page_hinkley.update(confidence)

        # ── Score composite pondéré ───────────────────────────────────
        signals = {
            'ADWIN':        float(self.adwin.drift_detected),
            'KSWIN':        float(self.kswin.drift_detected),
            'PageHinkley':  float(self.page_hinkley.drift_detected),
        }
        self.composite_score = sum(self.WEIGHTS[k] * v for k, v in signals.items())

        # ── Logique de décision ───────────────────────────────────────
        if self.composite_score >= DRIFT_THRESH:
            if not self.drift_active:
                self.drift_events.append({
                    'timestamp':     datetime.now(timezone.utc).isoformat(),
                    'message_index': self.messages_total,
                    'score':         round(self.composite_score, 4),
                    'signals':       signals,
                })
            self.drift_active    = True
            self.drift_confirmed = self.composite_score >= CONFIRM_THRESH
            self.messages_since_drift = 0
        else:
            self.messages_since_drift += 1
            if self.messages_since_drift > 1000:
                self.drift_active    = False
                self.drift_confirmed = False

        return {
            'drift':           self.drift_active,
            'composite_score': round(self.composite_score, 4),
            'signals':         signals,
            'recommended_lr':  LR_DRIFT if self.drift_active else LR_BASE,
        }

    def get_recommended_lr(self) -> float:
        return LR_DRIFT if self.drift_active else LR_BASE

    def is_drift_active(self) -> bool:
        return self.drift_active

    def get_alert_payload(self) -> dict:
        conf_window = self.confidence_history[-100:]
        return {
            'timestamp':              datetime.now(timezone.utc).isoformat(),
            'composite_score':        round(self.composite_score, 4),
            'signals': {
                'ADWIN':       bool(self.adwin.drift_detected),
                'KSWIN':       bool(self.kswin.drift_detected),
                'PageHinkley': bool(self.page_hinkley.drift_detected),
            },
            'drift_confirmed':        self.drift_confirmed,
            'recommended_lr':         self.get_recommended_lr(),
            'messages_total':         self.messages_total,
            'confidence_mean_last100': round(
                sum(conf_window) / len(conf_window), 4
            ) if conf_window else 0.0,
            'total_drift_events':     len(self.drift_events),
        }

    def get_stats(self) -> dict:
        return {
            'messages_total':    self.messages_total,
            'composite_score':   round(self.composite_score, 4),
            'drift_active':      self.drift_active,
            'drift_confirmed':   self.drift_confirmed,
            'total_drift_events': len(self.drift_events),
            'current_lr':        self.get_recommended_lr(),
        }
