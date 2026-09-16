# spark-app/src/drift_monitor.py — Tri-Détecteur de Concept Drift
# KOMOSSI Sosso — Master BIG DATA IA, Institut ESI — UCAO-UUT, 2025-2026
#
# Détecteurs River 0.21.2 disponibles : ADWIN, KSWIN, PageHinkley
# Note : EDDM n'est pas dans River 0.21.2 → remplacé par PageHinkley
#   - ADWIN      (poids 0.45) : dérive abrupte — fenêtre adaptative sur p_fake
#   - KSWIN      (poids 0.35) : dérive statistique — test de Kolmogorov-Smirnov sur p_fake
#   - PageHinkley(poids 0.20) : dérive graduelle — somme cumulée sur p_fake
#   - EntropyADWIN (canal indépendant, hors composite) : ADWIN sur l'entropie
#     binaire H(p_fake) — dérive de CONFIANCE, cf. verrou identifié dans
#     reports/EXPERIENCES_2026-08-29.md : une campagne de fake hors-distribution
#     (LIAR, p_fake moyen ~0,68 au lieu de ~0,97) ne déplace quasiment pas la
#     MOYENNE de p_fake (0,50 -> ~0,55, sous le seuil des 3 détecteurs
#     historiques) mais fait s'effondrer la confiance du modèle :
#     H(0,68) ≈ 0,90 contre H(0,05-0,10) ≈ 0,29-0,47 pour une prédiction confiante.
#     IMPORTANT — pourquoi ce canal n'est PAS mélangé au score pondéré : une
#     première version lui donnait un poids (0,25) prélevé sur ADWIN/KSWIN/
#     PageHinkley ; reports/tableau_4_3_tri_detecteur.md a montré que cela
#     DÉGRADE le TPR des scénarios B et D (0,27→0,20 et 0,20→0,17) car
#     EntropyADWIN ne se déclenche jamais sur les scénarios A-D (dérive entre
#     deux régimes confiants, l'entropie reste basse des deux côtés) : diluer
#     les poids des 3 détecteurs éprouvés pour un canal aveugle à ces
#     scénarios est une perte nette. EntropyADWIN déclenche donc désormais une
#     alerte de façon indépendante (OU logique), sans toucher au score
#     composite des 3 détecteurs historiques — best of both worlds : aucune
#     régression sur A-D, détection ajoutée sur le mode de dérive LIAR.

from river import drift
from typing import Dict
from datetime import datetime, timezone
import os, math


ADWIN_DELTA    = float(os.getenv('ADWIN_DELTA', 0.002))
KSWIN_WINDOW   = int(os.getenv('KSWIN_WINDOW_SIZE', 100))
KSWIN_ALPHA    = float(os.getenv('KSWIN_ALPHA', 0.005))
PH_DELTA       = float(os.getenv('PH_DELTA', 0.005))        # sensibilité PageHinkley
PH_THRESHOLD   = float(os.getenv('PH_THRESHOLD', 50.0))     # seuil PageHinkley
ENTROPY_ADWIN_DELTA = float(os.getenv('ENTROPY_ADWIN_DELTA', 0.002))
# Empiriquement, sur le bruit réel du modèle, ADWIN est le détecteur le plus
# fiable sur p_fake : il se déclenche de façon répétée et cohérente, mais
# KSWIN/PageHinkley se synchronisent rarement avec lui dans la même fenêtre de
# rémanence. Seuil abaissé de 0.5 à 0.4 pour qu'un seul signal fort (ADWIN sur
# p_fake OU ADWIN sur l'entropie) suffise à confirmer la dérive.
DRIFT_THRESH   = float(os.getenv('DRIFT_COMPOSITE_THRESHOLD', 0.4))
CONFIRM_THRESH = float(os.getenv('DRIFT_CONFIRMED_THRESHOLD', 0.8))
LR_BASE        = float(os.getenv('ONLINE_LR_BASE', 1e-5))
LR_DRIFT       = float(os.getenv('ONLINE_LR_DRIFT', 5e-5))


def binary_entropy(p: float) -> float:
    """Entropie de Shannon (base 2) d'une décision binaire, normalisée [0, 1].

    Copie locale de calibration.binary_entropy (drift_monitor.py doit rester
    utilisable sans dépendre de calibration.py — l'API démo n'importe pas Spark).
    """
    p = float(min(max(p, 1e-12), 1.0 - 1e-12))
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


class DynamicDriftMonitor:
    """
    Tri-Détecteur de Concept Drift (River 0.21.2) :
    - ADWIN        (poids 0.45) : surveillance de la moyenne de p_fake — dérive abrupte
    - KSWIN        (poids 0.35) : test statistique KS sur p_fake — changement de distribution
    - PageHinkley  (poids 0.20) : somme cumulée sur p_fake — dérive graduelle
    - EntropyADWIN (canal indépendant, OU logique) : ADWIN sur l'entropie H(p_fake) —
      dérive de CONFIANCE. Activable/désactivable via use_entropy_channel pour la
      comparaison A/B (scripts/experiment_tri_detecteur.py,
      scripts/experiment_derive_opportunite.py). Voir docstring de module pour la
      justification du choix OU plutôt que pondération.
    """
    WEIGHTS = {'ADWIN': 0.45, 'KSWIN': 0.35, 'PageHinkley': 0.20}

    def __init__(self, use_entropy_channel: bool = None):
        if use_entropy_channel is None:
            use_entropy_channel = os.getenv('DRIFT_ENTROPY_CHANNEL', 'true').lower() == 'true'
        self.use_entropy_channel = use_entropy_channel
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
        self.entropy_adwin = drift.ADWIN(delta=ENTROPY_ADWIN_DELTA) if use_entropy_channel else None
        self.weights = dict(self.WEIGHTS)
        self._names = list(self.weights.keys())

        self.composite_score       = 0.0
        self.entropy_active        = False
        self.drift_active          = False
        self.drift_confirmed       = False
        self.messages_total        = 0
        self.messages_since_drift  = 0
        self.drift_events          = []
        self.confidence_history    = []
        # Fenêtre de rémanence (en messages) : chaque détecteur ne pulse
        # 'drift_detected=True' que le message exact où il bascule (ADWIN,
        # KSWIN et PageHinkley se déclenchent presque toujours à des indices
        # différents). Sans rémanence, le score composite ne peut quasiment
        # jamais additionner deux signaux au même instant et ne franchit
        # donc jamais DRIFT_THRESH. On garde chaque signal actif pendant
        # SIGNAL_HOLD messages après son déclenchement pour permettre aux
        # 3 détecteurs de se combiner s'ils repèrent la même dérive à
        # quelques messages d'intervalle. EntropyADWIN suit la même
        # rémanence mais reste hors du score composite (voir docstring).
        self.SIGNAL_HOLD = int(os.getenv('DRIFT_SIGNAL_HOLD_MESSAGES', 100))
        self._hold        = {name: 0 for name in self._names}
        self._entropy_hold = 0
        print(
            f'[DRIFT] Monitor initialisé | '
            f'ADWIN δ={ADWIN_DELTA} | KSWIN α={KSWIN_ALPHA} | '
            f'PageHinkley δ={PH_DELTA} λ={PH_THRESHOLD} | '
            f'EntropyADWIN {"actif (OU logique) δ=" + str(ENTROPY_ADWIN_DELTA) if use_entropy_channel else "désactivé"}'
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

        # ── Mise à jour des 3 détecteurs (score composite) ────────────
        self.adwin.update(confidence)
        self.kswin.update(confidence)
        self.page_hinkley.update(confidence)
        detectors = [('ADWIN', self.adwin), ('KSWIN', self.kswin), ('PageHinkley', self.page_hinkley)]

        # ── Rémanence : un détecteur qui vient de pulser reste "actif"
        #    pendant SIGNAL_HOLD messages, pour laisser une chance aux
        #    deux autres détecteurs de le rejoindre dans le score composite.
        for name, detector in detectors:
            if detector.drift_detected:
                self._hold[name] = self.SIGNAL_HOLD
            elif self._hold[name] > 0:
                self._hold[name] -= 1

        # ── Score composite pondéré (3 détecteurs historiques, inchangé) ──
        signals = {name: float(self._hold[name] > 0) for name in self._names}
        self.composite_score = sum(self.weights[k] * v for k, v in signals.items())

        # ── Canal EntropyADWIN : indépendant, en OU logique ───────────
        if self.use_entropy_channel:
            self.entropy_adwin.update(binary_entropy(confidence))
            if self.entropy_adwin.drift_detected:
                self._entropy_hold = self.SIGNAL_HOLD
            elif self._entropy_hold > 0:
                self._entropy_hold -= 1
            self.entropy_active = self._entropy_hold > 0
            signals['EntropyADWIN'] = float(self.entropy_active)

        # ── Logique de décision : score composite >= seuil OU EntropyADWIN actif ──
        triggered = self.composite_score >= DRIFT_THRESH or self.entropy_active
        if triggered:
            if not self.drift_active:
                self.drift_events.append({
                    'timestamp':     datetime.now(timezone.utc).isoformat(),
                    'message_index': self.messages_total,
                    'score':         round(self.composite_score, 4),
                    'signals':       signals,
                    'entropy_triggered': self.entropy_active and self.composite_score < DRIFT_THRESH,
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
        signals = {name: self._hold[name] > 0 for name in self._names}
        if self.use_entropy_channel:
            signals['EntropyADWIN'] = self.entropy_active
        return {
            'timestamp':              datetime.now(timezone.utc).isoformat(),
            'composite_score':        round(self.composite_score, 4),
            'signals': signals,
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
