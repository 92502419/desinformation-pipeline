# Baselines statiques — comparaison avec Continual-DistilBERT (H2)

_Généré par `scripts/train_baselines.py` le 2026-09-15 — entraînées sur le même split (46660 train / 15554 test) que le modèle Transformer._

Répond à un écart identifié dans `reports/CONFORMITE_PROPOSITION.md` (H2, Tableau 4.1) : ces deux baselines (absentes du dépôt après l'incident du 26/08/2026) sont ici réellement ré-entraînées et évaluées sur `data/processed/test/test.csv`.

| Modèle | F1-macro | Précision | Rappel | AUC-ROC | Avg. Precision |
|---|---|---|---|---|---|
| TF-IDF + Régression Logistique | 0.9155 | 0.8976 | 0.9380 | 0.9759 | 0.9754 |
| TF-IDF + Linear SVM | 0.9178 | 0.9091 | 0.9285 | 0.9726 | 0.9714 |
| Continual-DistilBERT (référence, reports/metrics_summary.json) | 0.9370 | 0.9387 | 0.9351 | 0.9899 | 0.9900 |

**Écart Continual-DistilBERT vs meilleure baseline classique : +0.0192 point de F1-macro.**
