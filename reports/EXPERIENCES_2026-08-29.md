# Expériences reproduites le 2026-08-29 — tri-détecteur & « dérive = opportunité »

Ces deux expériences comblent les cases restées vides du mémoire après l'incident du
26/08/2026 (Tableaux 4.3 et 4.4, hypothèse H3). **Aucune donnée générée** : seuls des
articles réels du jeu de test (`data/processed/test/test.csv`) sont utilisés ; seul
l'ordre et la proportion fake/réel au cours du temps sont contrôlés, exactement comme
le décrit `scripts/inject_drift_simulation.py` et la section 3.4.3.

Matériel : CPU (le pilote NVML de la machine est cassé — `torch.cuda` désactivé par
sécurité, cf. incident du 26/08). Modèle : `models/pretrained/` (DistilBERT multilingue
fine-tuné, F1-macro test 0,9370). River 0.26.1 (mémoire cite 0.21.2 ; mêmes algorithmes).

---

## 1. Tableau 4.3 — Performances du tri-détecteur

`scripts/experiment_tri_detecteur.py` — 30 répétitions/scénario, signal surveillé =
`p_fake` du modèle, paramètres identiques à `spark-app/src/drift_monitor.py`
(ADWIN δ=0,002 ; KSWIN α=0,005, fenêtre 100 ; PageHinkley δ=0,005, λ=50 ;
composite pondéré 0,45/0,35/0,20, rémanence 100, seuil 0,4). Fenêtre de détection
500 messages.

| Scénario | Détecteur | Délai moyen (msg) | TPR | FPR |
|---|---|---|---|---|
| A — abrupt | ADWIN | 77 | 1,00 | 0,00 |
| A — abrupt | KSWIN | 42 | 1,00 | **0,87** |
| A — abrupt | PageHinkley | 108 | 1,00 | 0,00 |
| A — abrupt | **Composite** | 76 | 1,00 | 0,00 |
| B — graduel | ADWIN | 439 | **0,13** | 0,00 |
| B — graduel | KSWIN | 168 | 0,60 | 0,90 |
| B — graduel | PageHinkley | 418 | 0,47 | 0,00 |
| B — graduel | **Composite** | 449 | 0,27 | 0,00 |
| C — cyclique | ADWIN | 86 | 1,00 | 0,00 |
| C — cyclique | KSWIN | 42 | 0,97 | 0,63 |
| C — cyclique | PageHinkley | 117 | 1,00 | 0,00 |
| C — cyclique | **Composite** | 85 | 1,00 | 0,00 |
| D — incrémental | ADWIN | 407 | **0,07** | 0,00 |
| D — incrémental | KSWIN | 206 | 0,70 | 0,90 |
| D — incrémental | PageHinkley | 416 | 0,37 | 0,00 |
| D — incrémental | **Composite** | 416 | 0,20 | 0,00 |

Figure : `reports/figures/09_tri_detecteur_scenarios.png`. Détail : `reports/tableau_4_3_tri_detecteur.{md,json}`.

**Interprétation :**
- Le **score composite** (piloté par ADWIN, poids 0,45) est **précis** (FPR 0,00 partout)
  et **rapide sur dérive abrupte et cyclique** (~80 messages, TPR 1,00).
- Il est **peu sensible aux dérives graduelle et incrémentale** (TPR 0,20–0,27, délai
  > 400 messages) : cohérent avec la littérature (une rampe lente est difficile à
  distinguer du bruit tôt) et avec le commentaire de `drift_monitor.py` (ADWIN se
  déclenche seul, KSWIN/PageHinkley se synchronisent rarement dans la même fenêtre).
- **KSWIN** est le plus rapide mais **génère beaucoup de fausses alarmes** sur le
  signal `p_fake` brut (FPR 0,63–0,90) : à ne pas utiliser seul.
- L'affirmation antérieure « PageHinkley le plus rapide sur A » est **fausse**
  (PageHinkley est le plus lent sur A : 108 vs ADWIN 77 vs KSWIN 42).

---

## 2. Tableau 4.4 / H3 — « la dérive comme opportunité d'apprentissage »

`scripts/experiment_derive_opportunite.py` — protocole **prequential** (test-then-train,
section 3.5.1). Flux : 800 messages stationnaires (welfake/fnn, 50 % fake) puis
**campagne abrupte** à t=800 (80 % de fake tirés du sous-corpus **LIAR** — déclarations
politiques courtes, terrain faible du modèle : dérive de **proportion** ET de
**concept**). Bras dynamique = calque 1:1 de `spark-app/src/nlp_classifier.py::online_update`
(reservoir équilibré 2 500/classe, replay 2/classe, batch 4+4, SGD momentum 0,9,
wd 0,01, clip 1,0, LR piloté par le tri-détecteur). Sonde d'oubli : 200 articles
welfake/fnn tenus à l'écart du flux, évalués périodiquement.

| Phase / mesure | F1 STATIQUE | F1 DYNAMIQUE (config prod) | F1 DYNAMIQUE (LR forcé 5e-5) |
|---|---|---|---|
| P0 — pré-campagne (welfake/fnn) | 0,970 | 0,970 | 0,970 |
| P1 — début de campagne | 0,712 | 0,712 | 0,712 |
| P2 — campagne établie (après adaptation) | 0,718 | **0,722** | **0,729** |
| Sonde d'oubli welfake/fnn (fin de flux) | 0,980 | **0,980** | **0,980** |
| Gain net dynamique en P2 | — | +0,4 pt | +1,1 pt |
| Oubli catastrophique | — | **0,0 pt** | **0,0 pt** |

Figures : `reports/figures/10_derive_opportunite.png` (config prod),
`10_derive_opportunite_oracle.png` (LR forcé). Détail :
`reports/experience_derive_opportunite{,_oracle}.{md,json}`.

**Résultat : H3 n'est PAS confirmée dans les conditions du pipeline.**
- Le modèle dynamique **ne récupère pas** son niveau pré-dérive après la campagne
  (−24 points de F1, comme le statique).
- Il **ne dépasse le statique que de +0,4 à +1,1 point** en P2 — dans le bruit.
- Le « +21,8 points » du mémoire v7 **n'est pas reproductible**.
- Le tri-détecteur **n'a pas déclenché** sur cette campagne : les fake LIAR sont
  classés avec un `p_fake` faible (~0,68 au lieu de ~0,97), la moyenne du signal ne
  monte que de 0,50 à ~0,55 — sous le seuil des détecteurs.

**Ce qui EST démontré (résultat positif réel) :**
- **Aucun oubli catastrophique** : la sonde welfake/fnn reste à 0,980 tout au long du
  flux, dans les deux configurations. Le **reservoir replay équilibré remplit son rôle**.

**Deux verrous opérationnels identifiés (contribution honnête + travaux futurs) :**
1. **Signal de drift** : `p_fake` est peu sensible aux campagnes de fake
   hors-distribution (celles qui dégradent réellement le F1). Piste : signal basé sur
   le **taux d'erreur** (nécessite un flux de labels, même partiel/différé) ou sur la
   **dérive des embeddings** en amont du classifieur.
2. **Budget d'adaptation** : LR 1e-5 et ~8 exemples/mise à jour sont trop faibles pour
   ré-apprendre un domaine à un modèle de 135 M paramètres. Piste : LR plus agressif
   pendant la dérive **sous garde-fou anti-oubli** (le reservoir replay, ici validé, le
   permet).

---

## Reproduction

```bash
source venv_main/bin/activate
python scripts/experiment_tri_detecteur.py --reps 30
python scripts/experiment_derive_opportunite.py                       # config production
python scripts/experiment_derive_opportunite.py --force_lr 5e-5 --tag oracle
```
