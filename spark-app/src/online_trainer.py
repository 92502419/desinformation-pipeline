# spark-app/src/online_trainer.py — Module de gestion de l'entraînement online
# Utilisé par spark_streaming.py pour orchestrer le cycle d'apprentissage continu
import logging
import os

log = logging.getLogger(__name__)

SYNC_EVERY_N_BATCHES = int(os.getenv('SYNC_ONNX_EVERY', 100))


class OnlineTrainer:
    """
    Orchestre l'entraînement continu :
    - Accumule les nouvelles données
    - Déclenche les mises à jour de gradient via nlp_classifier
    - Synchronise le modèle ONNX périodiquement
    """

    def __init__(self, nlp_classifier, drift_monitor):
        self.nlp = nlp_classifier
        self.drift = drift_monitor
        self.batch_count = 0
        self.total_examples = 0
        self.total_loss = 0.0

    def step(self, batch_texts: list, batch_labels: list) -> dict:
        """
        Effectue un pas d'entraînement online sur un batch.
        Retourne les métriques du pas.
        """
        if not batch_texts:
            return {'loss': 0.0, 'lr': self.drift.get_recommended_lr(), 'batch': self.batch_count}

        lr = self.drift.get_recommended_lr()
        loss = self.nlp.online_update(batch_texts, batch_labels, lr=lr)

        self.batch_count += 1
        self.total_examples += len(batch_texts)
        self.total_loss += loss

        log.info(
            f'[OnlineTrainer] Batch {self.batch_count} | '
            f'loss={loss:.4f} | lr={lr:.2e} | '
            f'examples={self.total_examples} | '
            f'drift_active={self.drift.is_drift_active()}'
        )

        # Synchronisation ONNX périodique
        if self.batch_count % SYNC_EVERY_N_BATCHES == 0:
            log.info(f'[OnlineTrainer] Synchronisation ONNX (batch {self.batch_count})...')
            try:
                self.nlp.sync_onnx()
                log.info('[OnlineTrainer] Modèle ONNX resynchronisé.')
            except Exception as e:
                log.warning(f'[OnlineTrainer] Erreur sync ONNX (non bloquant) : {e}')

        return {
            'loss': loss,
            'lr': lr,
            'batch': self.batch_count,
            'total_examples': self.total_examples,
            'avg_loss': self.total_loss / self.batch_count,
        }

    def get_stats(self) -> dict:
        return {
            'batch_count': self.batch_count,
            'total_examples': self.total_examples,
            'avg_loss': self.total_loss / max(self.batch_count, 1),
        }
