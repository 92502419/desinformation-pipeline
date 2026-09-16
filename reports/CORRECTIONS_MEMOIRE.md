# Corrections à apporter au mémoire — mise en conformité avec le dépôt réel

_État : 2026-08-29. Basé sur une relecture ligne par ligne de
`Memoire_Master2_IBDIA_KOMOSSI_Sosso_v7.docx` confrontée aux fichiers du dépôt
(`data/processed/preprocessing_report.json`, `reports/metrics_summary.json`,
`producer/src/`, `spark-app/src/`, `.env.example`, `config/grafana/`, `docker-compose.yml`)._

**Aucune donnée n'a été inventée.** Toutes les valeurs « corriger en » ci-dessous
proviennent d'un fichier ou d'un script du dépôt, cité entre parenthèses.

Convention : **[REMPL]** = remplacer un chiffre/mot ; **[SUPPR]** = supprimer une
affirmation non réalisée ; **[REQUAL]** = requalifier un résultat comme « attendu /
non revérifié après l'incident » ; **[HARMO]** = incohérence interne à harmoniser.

---

## A. Nom de la formation (partout)

| Où | Écrit actuellement | Corriger en |
|---|---|---|
| Page de garde, en-têtes, §1.6.1, §6.1, nom de fichier | « Master 2 IBDIA », « Ingénierie Big Data et Intelligence Artificielle », « IBDIA » | **« Master BIG DATA IA »**, École Supérieure d'Ingénieurs (**Institut ESI**), UCAO-UUT. Ne plus employer l'acronyme « IBDIA ». |
| §6.1 vs §1.6.1 | §6.1 dit « Master BIG DATA IA », §1.6.1 dit « Master 2 en Ingénierie Big Data et Intelligence Artificielle » | **[HARMO]** un seul libellé : « Master BIG DATA IA, Institut ESI — UCAO-UUT ». |

> Fait dans le code : `README.md`, `streamlit-dashboard/app.py`, `api/src/main.py`,
> `spark-app/src/drift_monitor.py` utilisent désormais « Master BIG DATA IA, Institut ESI — UCAO UUT ».

---

## B. Corpus d'entraînement — §Résumé, §Abstract, §3.1.1, §3.1.3, §6.1

Source de vérité : `data/processed/preprocessing_report.json` + comptage des fichiers bruts.

### B.1 — Volume total

| Où | Écrit | Corriger en | Justification |
|---|---|---|---|
| Résumé | « corpus de ~77 768 exemples (après fusion et déduplication des 4 datasets) — agrégant … ISOT (44 898), WELFake (72 134), LIAR (12 836), FakeNewsNet (23 196) » | garder **77 768 après déduplication** ; ajouter **174 687 agrégés avant déduplication** | `preprocessing_report.json` : `total_before_dedup` n'est pas stocké tel quel mais la somme réelle est 44 898 + 72 134 + 23 196 + 12 836 (index) + 21 623 (africain) = **174 687** ; `total_after_dedup = 77 768` ; `duplicates_removed = 96 291` |
| **Abstract** | « pre-trained on a corpus of **~153,064 examples** aggregating four public datasets » | **[HARMO]** « ~174,687 aggregated examples … reduced to **77,768 unique** after strict title deduplication » — **le Résumé (FR) et l'Abstract (EN) se contredisent** (77 768 vs 153 064) | idem |
| §3.1.1 | « quatre datasets … corpus d'entraînement de **~153 000 exemples** » ET plus loin « **~77 768 exemples** » ET « **Deux datasets publics sont utilisés** » | **[HARMO]** une seule formulation : « 5 sources agrégées (4 datasets classiques + 1 sous-corpus africain) = 174 687 exemples bruts → 77 768 uniques après déduplication » | idem ; la phrase « Deux datasets publics sont utilisés » est un reliquat, à supprimer |
| §3.1.3, §6.1 | « corpus fusionné de ~77 768 exemples » | **correct** — conserver, mais s'assurer que Résumé/Abstract disent la même chose |

### B.2 — ISOT

| Où | Écrit | Corriger en |
|---|---|---|
| Résumé, Abstract, §3.1.1, §6.1 | ISOT listé comme 4ᵉ dataset contribuant au corpus (44 898) | **[REMPL]** préciser : « ISOT (44 898 articles) — **entièrement absorbé par déduplication** : ses articles Reuters + sites fake sont déjà présents dans WELFake (recoupement connu). ISOT ne contribue à **aucun** exemple unique au corpus final. » (`preprocessing_report.json` : la clé `isot` est **absente** de `sources`) |

### B.3 — Comptes par dataset (après déduplication, corpus final de 77 768)

| Dataset | Mémoire | Réel (`preprocessing_report.json` → `sources`) |
|---|---|---|
| WELFake | 72 134 (brut) | **44 609** dans le corpus final |
| FakeNewsNet | 23 196 | **13 350** — FakeNewsNet ne distribue que des **index** (URL + métadonnées), une grande partie des corps d'article sont morts/trop courts ; **[REMPL]** « 23 196 lignes d'index, ~13 350 avec texte exploitable » |
| LIAR | 12 836 | **9 247** après passage 6 classes → binaire et nettoyage |

### B.4 — Sous-corpus africain (§3.1.1) — **écart majeur**

| Où | Écrit | Corriger en | Justification |
|---|---|---|---|
| §3.1.1 | « sous-corpus africain porté de **854 à 3 601** exemples grâce à **2 742** articles MasakhaNEWS couvrant **11 langues** » | **[REMPL]** « sous-corpus africain de **21 623** exemples (`data/raw/africa_news/africa_news.csv`) : **~21 400 articles MasakhaNEWS** couvrant **12 langues** (amharique, haoussa, igbo, lingala, oromo, pidgin nigérian, **kirundi**, shona, somali, swahili, tigrinya, yoruba) + 207 articles Google News RSS africain. Après équilibrage 50/50, **10 562** exemples africains subsistent dans le corpus final. » | `africa_news.csv` : 21 623 lignes, `source.value_counts()` = 12 codes `masakhanews_*` + `google_news_africa_rss` ; `preprocessing_report.json` : somme `masakhanews_* = 10 466` + `google_news_africa_rss = 96` = 10 562 |
| §3.1.1 | « le volume de fake news africaines authentiques reste limité (~77 exemples) » | **[REMPL]** « **le corpus africain ne contient aucun exemple `fake`** : les 21 623 articles MasakhaNEWS + RSS sont tous des dépêches de presse vérifiées (label = 0). Africa Check inaccessible (Cloudflare 403). » (`africa_news.csv` : `label.value_counts()` = `{0: 21623}`) |
| Résumé, Abstract, §3.1.1 | « MasakhaNEWS à **11 langues** » | **[REMPL]** partout : **12 langues** |

### B.5 — Fakeddit (Abstract, §3.1.1)

| Où | Écrit | Statut |
|---|---|---|
| Abstract (« Fakeddit was removed … 2025 ») et §3.1.1 (note) | Fakeddit retiré, inaccessible | **Correct** — conserver. Reformuler l'insertion maladroite en milieu de phrase de l'Abstract. |

### B.6 — Distribution de classes (§3.1.3)

| Écrit | Vérifiable ? |
|---|---|
| « corpus brut à **58,7 % fake / 41,3 % réel** (ratio 1,42:1) » | Non recalculé après l'incident — `preprocessing_report.json` ne stocke pas la répartition avant équilibrage. **[REQUAL]** ou recalculer via un petit script sur `data/raw/`. |
| « VADER moyen +0,41 vs +0,06 ; exclamation ×3,2 ; marques d'exclusivité ratio 4,7:1 » | Statistiques d'EDA d'avant l'incident, **non reproduites** (pas de notebook/script d'EDA dans le dépôt). **[REQUAL]** « analyse exploratoire réalisée sur une version antérieure du corpus » ou re-générer. |
| « couvre **2010–2024** » vs « 2011–2024 » ailleurs | **[HARMO]** un seul intervalle. |

---

## C. Flux de données temps réel — §Résumé, §Abstract, §3.1.2, §3.3.2

Source : `producer/src/rss_sources.py`, `producer/src/gdelt_client.py`, `producer/src/kafka_producer.py`.

| Où | Écrit | Corriger en | Justification |
|---|---|---|---|
| §3.1.2, proposition | « **60 sources** RSS (40 mainstream + 20 douteuses identifiées par AFP Factuel, Africa Check, Snopes) » | **[REMPL]** « **12 flux RSS**, tous de la catégorie *fiable* (AFP, Reuters, BBC, Al Jazeera, Le Monde, RFI, AP News, Deutsche Welle, France24, VOA, Jeune Afrique, Africa News) ; `RSS_SOURCES['suspicious']` est **vide** — l'ajout de sources douteuses est une piste de travaux futurs. » | `rss_sources.py` : `reliable` = 12 entrées, `suspicious = []` |
| §3.1.2, Résumé, Abstract | « **API X (ex-Twitter) Academic — niveau gratuit**, 500 000 tweets/mois » | **[SUPPR]** entièrement. Le niveau *Academic Research* gratuit de l'API X a été fermé en 2023 ; aucun code Twitter/X dans `producer/`. Remplacer par : « L'ingestion temps réel repose sur **deux** mécanismes : flux RSS + API GDELT DOC 2.0. » | `grep -ri twitter\|tweepy producer/` → aucun résultat |
| §3.1.2 | « API GDELT **GKG** (Global Knowledge Graph)… entités nommées, thèmes, tonalité » | **[REMPL]** « API GDELT **DOC 2.0** (`https://api.gdeltproject.org/api/v2/doc/doc`), requête sur le thème `FAKE_NEWS,DISINFORMATION`, champ `tone` récupéré (`gdelt_tone`). Le GKG complet (entités/thèmes) n'est pas exploité. » | `gdelt_client.py` |
| §3.3.2, §1.4, Résumé | « micro-batch Spark **2 secondes** » / « micro-batches (2 s) » | **[REMPL]** **5 secondes** partout | `.env.example` : `SPARK_MICRO_BATCH_INTERVAL=5 seconds` ; `spark_streaming.py` lit cette variable ; §3.3.3 dit déjà « 5 secondes » — **[HARMO]** |
| §3.3.2 | « **Trois topics Kafka** sont définis » + Tableau 3.4 | **[REMPL]** **Deux topics** : `raw-news-stream` (`KAFKA_TOPIC_RAW`) et `drift-alerts` (`KAFKA_TOPIC_DRIFT`). Supprimer le 3ᵉ topic (`classified-news`) du Tableau 3.4 et de la Figure d'architecture — Spark écrit directement dans MongoDB/Elasticsearch. | `grep -rn KAFKA_TOPIC spark-app/ api/ producer/` → seulement `_RAW` et `_DRIFT` |
| §3.3.2 | « scrape les flux RSS toutes les **60 secondes** et interroge GDELT toutes les **15 minutes** » | **Correct** (`RSS_SCRAPE_INTERVAL_SEC=60`, `GDELT_QUERY_INTERVAL_SEC=900`). Conserver. |
| §3.3.2 | « TTL 7 200 s / 2 heures » | **Correct** (`RSS_DEDUP_TTL_SEC=7200`). Conserver. |

---

## D. Modèle NLP & résultats — §Résumé, §Abstract, §4.1.1, §4.1.3, §4.4, §6.1

Source : `reports/metrics_summary.json`, `reports/RAPPORT_ANALYSE.md`.

> **MISE À JOUR 2026-08-29 — les expériences manquantes ont été exécutées.**
> Tableau 4.3 (tri-détecteur) et Tableau 4.4 / H3 (« dérive = opportunité ») sont
> désormais produits sur données réelles : voir `reports/EXPERIENCES_2026-08-29.md`,
> `reports/tableau_4_3_tri_detecteur.md`, `reports/experience_derive_opportunite.md`.
> Ces résultats sont déjà intégrés dans **`Memoire_..._v8.docx`** (Tableaux 4.3/4.4
> réécrits, §4.1.2 / §4.1.3 / §4.4 SQ3 / §6.1 requalifiés). **Verdict : H3 n'est PAS
> confirmée dans les conditions du pipeline** (gain dynamique +0,4 à +1,1 pt, dans le
> bruit ; pas de récupération) ; en revanche **aucun oubli catastrophique** (reservoir
> replay validé). Le « +21,8 pts » de v7 est infirmé. Les points D.3, D.4, D.5
> ci-dessous restent donc « à requalifier » plutôt que « à produire » — c'est fait
> dans v8, garder ces lignes comme trace de ce qui a changé.

### D.1 — Métriques finales (déjà en grande partie corrigées dans v7 — vérifier partout)

| Métrique | Valeur correcte | Où re-vérifier |
|---|---|---|
| F1-macro (test) | **0,9370** (93,70 %) | Résumé ✓, Abstract ✓, §4.1.1 ✓ — mais §6.1 dit « F1=0.932 » **[HARMO → 0,9370]** |
| AUC-ROC (test) | **0,9899** | OK dans v7 |
| Average Precision | **0,9900** | OK |
| Précision / Rappel | **0,9387 / 0,9351** | OK |
| F1 validation (meilleure époque) | **0,9392** (époque **3/10**) | OK |
| Latence ONNX INT8 | **19,16 ms/article** (mesurée, 100 itérations, CPU dev) | Résumé ✓ ; **§4.2.3 dit encore « moins de 10 ms/article »** → **[REMPL]** ou préciser « ~10 ms en lot dans l'API, 19,16 ms en benchmark unitaire » |
| Matrice de confusion (test) | **TP 7272 · TN 7302 · FP 475 · FN 505** | OK dans v7 §4.1.1 |

### D.2 — Taille du jeu de test (§4.1.1) — **[HARMO]**

| Où | Écrit | Corriger en |
|---|---|---|
| §4.1.1, para 3 | « La matrice de confusion sur le jeu de test **complet (116 917 exemples, 10 % du corpus)** » | **15 554 exemples** (`preprocessing_report.json` : `n_test = 15554`). Supprimer « 116 917 » et « 10 % du corpus ». Le paragraphe suivant dit déjà « 15 554 » → contradiction interne dans la même sous-section. |

### D.3 — Comparaison aux baselines (§4.1.1 Tableau 4.1, §4.4 SQ2, §6.1) — **NON RÉALISÉE**

| Où | Écrit | Correction |
|---|---|---|
| §4.1.1 | « comparés à **quatre baselines statiques** et au modèle statique DistilBERT … Tableau 4.1 » | **[REQUAL]** `scripts/train_model.py` n'entraîne **que** le DistilBERT. Aucune baseline (TF-IDF+LR, TF-IDF+RF, BiLSTM+GloVe, BERT-base, RoBERTa-base, DistilBERT figé) n'a été ré-entraînée après l'incident. Deux options : **(a)** produire réellement au minimum 2 baselines (TF-IDF+régression logistique, un LSTM) sur le même split `data/processed/` — faisable en < 1 h CPU ; **(b)** marquer le Tableau 4.1 « valeurs indicatives issues de la littérature / d'un run antérieur non reproductible ». |
| §4.4 SQ2, §6.1 | « Continual-DistilBERT (F1=0.932) **surpasse RoBERTa statique (F1=0.921)** » | **[SUPPR ou REQUAL]** RoBERTa n'a jamais été entraîné dans ce projet. |

### D.4 — « Dérive = opportunité » : gain dynamique vs statique (§4.1.3 Tab. 4.4, §4.4 SQ3, §6.1) — **CONTRIBUTION PRINCIPALE NON DÉMONTRÉE**

| Où | Écrit | Correction |
|---|---|---|
| §4.4 SQ3 | « gain net **+21,8 points de F1** en dérive graduelle à 3 000 messages post-dérive (objectif : 8) » | **[REQUAL / à produire]** Aucun run comparatif Continual vs statique figé n'existe dans le dépôt (`reports/` ne contient que l'entraînement hors-ligne). C'est **le résultat central du mémoire** : il faut soit exécuter l'expérience (geler une copie du modèle époque 3, rejouer le scénario B via `scripts/inject_drift_simulation.py` sur les deux, comparer le F1 sur une fenêtre glissante), soit indiquer explicitement « résultat attendu, non revérifié lors de la reprise ». |
| §1.3.2 RA 4.2 | « F1 dynamique post-adaptation > F1 baseline pré-dérive de **+0.9 point ou plus** » | idem — cible, pas résultat mesuré. |
| §6.1 « Contribution expérimentale majeure » | reformule le +0.9 / +21.8 comme acquis | idem [REQUAL] tant que l'expérience n'est pas faite. |

### D.5 — Tri-détecteur : Tableau 4.3 (§4.1.2) — 12 mesures non régénérées

« ADWIN, KSWIN, PageHinkley … évalués sur les quatre scénarios … délai de détection en nombre de messages » — **[REQUAL]** aucun script ne produit ce tableau (3 algos × 4 scénarios : délai, TPR, FPR). `scripts/inject_drift_simulation.py` injecte les scénarios mais ne journalise pas ces métriques comparatives. À instrumenter, ou à requalifier.

---

## E. Seuils & configuration du drift — §3.4.2, §3.4.3, §4.x

Source : `spark-app/src/drift_monitor.py`, `.env.example`.

| Où | Écrit | Corriger en | Justification |
|---|---|---|---|
| §3.4.2 Tab. 3.6, dashboards | « seuil de **détection 0,5** » | **seuil d'alerte 0,4** (`DRIFT_COMPOSITE_THRESHOLD=0.4`) — abaissé de 0,5 à 0,4 pour qu'ADWIN seul (poids 0,45) suffise à déclencher | `drift_monitor.py` l.16-26 + commentaire explicatif |
| §3.4.2 | seuil de confirmation | **0,8** (`DRIFT_CONFIRMED_THRESHOLD=0.8`) — inchangé, correct |
| §3.4.2 | poids | ADWIN **0,45** / KSWIN **0,35** / PageHinkley **0,20** — correct |
| §3.4.2 | LR adaptatif | **1e-5** (stable, `ONLINE_LR_BASE`) → **5e-5** (dérive, `ONLINE_LR_DRIFT`) — vérifier que le mémoire ne cite pas d'autres valeurs |
| §3.4.3 Scénario B | « proportion fake **0 % → 80 %** sur 3 000 articles » | **[HARMO]** `scripts/inject_drift_simulation.py` (scénario B) + dashboard : **50 % → 90 %**. Aligner le texte sur le script, ou le script sur le texte. |
| §3.4.3 | « drift-injector … toutes les 30 minutes … désactivé par défaut », `visualization_window` 60-600 s (300 s défaut) | **Correct** (`api/src/routers/drift.py`). Conserver. |
| §3.4.3 NB | « taux de fake résiduel réel ~52 % après purge » | **[REMPL]** valeur datée. `GET /api/v1/stats` au 2026-08-29 : `total_articles=303`, `fake_rate=13.2` (bases ré-initialisées depuis). Donner la valeur du jour de la soutenance, ou retirer le chiffre. |

---

## F. Tableau de bord Grafana — §4.2.1 — **écart important**

Source : `config/grafana/dashboards/pipeline-desinformation.json`, `config/grafana/provisioning/`.

| Où | Écrit | Corriger en | Justification |
|---|---|---|---|
| §4.2.1 | « **treize panneaux**, dont six indicateurs de synthèse » | **7 panneaux** (P1 à P7) | le JSON provisionné contient exactement 7 objets `panels` |
| §4.2.1 | liste des 6 panneaux : *P2 jauge circulaire par source*, *P5 wordcloud des termes émergents*, *P6 heatmap géographique*, *P4 latence/throughput/taux d'erreur Spark* | **[REMPL]** par les panneaux réels : **P1** taux de fake (jauge, `avg(is_fake)`), **P2** flux réel vs fake par minute (`count`), **P3** score composite de drift (`avg(drift_score)`, seuils 0,4/0,8), **P4** table des évènements `drift_confirmed:true`, **P5** learning-rate adaptatif (`avg(recommended_lr)`), **P6** confiance moyenne du modèle (`avg(confidence)`), **P7** total d'articles traités (`count`). Aucun wordcloud, aucune heatmap géographique, aucune jauge par source. |
| §4.2.1 | « plage temporelle par défaut **24 heures** » | **6 heures** (`"time": {"from": "now-6h"}`) |
| §4.2.1 | « rafraîchissement automatique **30 secondes** » | **10 secondes** (`"refresh": "10s"`) |
| §4.2.1 | « **Cinq règles d'alerte** sont actives : Concept Drift détecté, Drift confirmé, Taux fake > 70 %, Taux fake > 40 %, Confiance < 70 % » | **[REMPL / à produire]** Aucune règle d'alerte n'est provisionnée dans Grafana (`config/grafana/provisioning/alerting/` n'existe pas). Ces 5 règles **sont** évaluées et affichées dans la page **Alertes** du dashboard **Streamlit**. Deux options : **(a)** ajouter `config/grafana/provisioning/alerting/alerts.yml` avec ces 5 règles ; **(b)** écrire « cinq règles d'alerte, évaluées dans le tableau de bord Streamlit (page Alertes) ; les évènements de drift sont par ailleurs persistés dans MongoDB, indexés dans Elasticsearch et publiés sur le topic Kafka `drift-alerts` ». |
| §1.3.2 RA 5.1 | « Dashboard Grafana … **7 panneaux actifs** » | **Correct** — conserver (cohérent avec le JSON). |
| §1.3.2 RA 5.1 | « Dashboard Streamlit … **5 pages** (Vue d'ensemble, Explorer les articles, Monitoring Drift, Analyse des sources, Configuration) » | **[REMPL]** **7 pages** : Tableau de bord, Articles temps réel, Recherche & Analyse, Drift & Apprentissage, Alertes, Infrastructure, À propos. Auto-refresh 30 s (configurable). |

---

## G. Interface Streamlit — §4.2.3

| Où | Écrit | Statut |
|---|---|---|
| §4.2.3 | page Recherche à 2 onglets (Web direct + Base de données ES), Google News RSS primaire + DuckDuckGo repli, badges FAKE/RÉEL, graphe de probabilités, export CSV, sujets suggérés Afrique | **Correct**, conforme à `api/src/routers/web_search.py` et à la page Recherche du dashboard. Conserver. |
| §4.2.3 | « endpoint `GET /api/v1/search/web` (+ `GET /api/v1/search/web/sources`) » | **Correct** (`web_search.py`). |
| §4.2.3 | « onnxruntime 1.19.2, transformers 4.44.2, tokenizers 0.19.1 » | Vérifier dans `api/requirements.txt` (l'environnement dev tourne en versions récentes non épinglées — voir README). |
| §4.2.3 | « mémoire du container API portée de 256 MB à **768 MB** » | **Correct** (`docker-compose.yml` : `api … memory: 768M`). |

---

## H. Architecture Docker — §3.3, §3.6, §1.4, §1.3.2

| Où | Écrit | Corriger en | Justification |
|---|---|---|---|
| §1.3.2 RA1, §3.3, résumés | « **11 services Docker** » | **13 services** : zookeeper, kafka, kafdrop, spark-master, spark-worker-1, spark-worker-2, mongodb, elasticsearch, grafana, rss-producer, spark-app, api, streamlit | `docker-compose.yml` |
| §1.4, §3.6 | « Apache Spark **3.5** » | **3.5.6**, image `bitnamilegacy/spark:3.5.6` (le namespace `bitnami/spark` a été déplacé vers `bitnamilegacy` mi-2025) | `docker-compose.yml` |
| §3.3.3 | « machine de référence (**12 GB RAM**) » | **14 Go de RAM** | README + `disinfo-pipeline-recovery` ; le mémoire mélange 12 et 14 Go |
| §3.3.3 | limite container Spark **3,5 GB**, conso 2,28 GiB | **Correct** (`spark-app … memory: 3584M` ≈ 3,5 Go) |
| §3.3.3 | G1GC, `maxOffsetsPerTrigger=200`, `ONLINE_TRAIN_EVERY_N=5` | **Correct** — conserver |
| §1.4 | « Python **3.12+** » | dev = **3.14** (pas de `python3.12-venv` installable sans root), conteneurs = **3.12** | README section 1 |
| Diagramme d'architecture (Figure) | topic `classified-news` ; « Spark 3.5.3 » ; « ~5-6 ms/article » | supprimer `classified-news` ; « Spark 3.5.6 » ; « ~19 ms/article (mesuré) » | idem C + D.1 |

---

## I. Résultats attendus §1.3.2 (RA1–RA6) — état réel

| # | Affirmation | État | Action doc |
|---|---|---|---|
| RA1 | architecture validée | config validée (`docker compose config`), **pas de smoke-test end-to-end documenté** | [REQUAL] « validée au niveau configuration ; test de bout en bout à rejouer » |
| RA2 | F1 ≥ cible | **0,9370 réel** | garder les vrais chiffres |
| RA3 | tri-détecteur calibré **et comparé (12 mesures)** | calibré ✓ ; **tableau comparatif non régénéré** | [REQUAL] ou produire |
| RA3.4 | alertes 4 canaux (Mongo, ES, Grafana P4, Kafka) | Mongo ✓, ES ✓, Kafka ✓, Grafana **panneau P4 ✓ mais pas de règle d'alerte** | voir F |
| RA4 | « dérive = opportunité », gain ≥ +0,9 pt | **non démontré** | [REQUAL] / à produire (priorité) |
| RA5.1 | Grafana 7 panneaux ✓ ; **3 règles d'alerte** ; provisioning rechargé < 2 min ; Streamlit **5 pages** | 7 panneaux ✓ ; **0 règle Grafana** ; Streamlit **7 pages** | corriger « 5 pages » → 7 ; « 3 règles » (RA5.1) ≠ « 5 règles » (§4.2.1) **[HARMO]** ; ajouter les règles ou requalifier |
| RA6.1 | déployable < 2 h, checklist 11 services | 13 services ; checklist non fournie | [REQUAL] + « 13 services » |
| RA6.3 | Tableau 4.3 : latence médiane, P95, throughput, lag max, uptime 2 h | **non renseigné** | à mesurer (2 h de run + relevé) ou [REQUAL] |
| RA6.4 | dépôt public **MIT** ; **3 notebooks Jupyter** | dépôt public ✓ ; **pas de fichier `LICENSE`** ; **notebooks/ = 1 README stub** | ajouter `LICENSE` MIT ; reconstruire 3 notebooks ou retirer la mention |

---

## J. Divers / cohérence rédactionnelle

| Où | Point | Action |
|---|---|---|
| Abstract | insertion « — Fakeddit was removed … — » au milieu d'une phrase | reformuler en phrase autonome |
| §1.4 | « HuggingFace Transformers **4.40** » vs §4.2.3 « 4.44.2 » | [HARMO] |
| Mots-clés / Keywords | inclut « WELFake » deux fois | nettoyer |
| §2.x, §3.1.3 | intervalle temporel du corpus « 2010–2024 » / « 2011–2024 » | [HARMO] |

---

## Récapitulatif : ce qui exige de PRODUIRE des données réelles (pas de rédaction seule)

1. **Expérience « dérive = opportunité »** (RA4 / §4.1.3 / §4.4 / §6.1) — comparaison
   Continual-DistilBERT vs modèle figé sur le scénario B. **Priorité 1.**
2. **Tableau 4.3 du tri-détecteur** (§4.1.2) — 3 algos × 4 scénarios (délai, TPR, FPR).
3. **Baselines** (§4.1.1 Tab. 4.1) — au moins TF-IDF+LR et un LSTM sur `data/processed/`.
4. **Métriques opérationnelles** (RA6.3) — latence médiane / P95, throughput, lag, uptime 2 h.
5. **3 notebooks Jupyter** (RA6.4) — perdus à l'incident.
6. **Smoke-test end-to-end des 13 services** + checklist (RA1 / RA6.1).
7. **`LICENSE` MIT** à la racine.
8. *(optionnel)* Règles d'alerte Grafana provisionnées (§4.2.1) ; flux RSS de sources douteuses (§3.1.2).
9. *(optionnel)* Re-générer l'EDA (§3.1.3 : VADER, ponctuation, ratio d'exclusivité, répartition 58,7/41,3).

## Ce qui se corrige uniquement dans le texte (le dépôt fait foi)

Toutes les lignes marquées **[REMPL]** / **[SUPPR]** / **[HARMO]** ci-dessus — principalement :
corpus 77 768 (et non 153 064), ISOT absorbé, africain 21 623 / 12 langues / 0 fake,
micro-batch 5 s, 2 topics Kafka, pas d'API X, 12 → RSS, seuil drift 0,4, test 15 554,
Grafana 7 panneaux (contenu réel) / refresh 10 s / plage 6 h, Streamlit 7 pages,
13 services, Spark 3.5.6, 14 Go RAM, latence 19,16 ms.
