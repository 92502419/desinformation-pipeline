# Vérification de conformité — Proposition de sujet ↔ état réel du projet

_Généré le 2026-08-28. Compare `Proposition_Sujet_Memoire_KOMOSSI_Sosso.docx` (et les
sections correspondantes de `Memoire_Master2_IBDIA_KOMOSSI_Sosso_v7.docx`) à ce qui est
**réellement présent et vérifiable** dans le dépôt après la reprise post-incident du
26/08/2026._

Légende : ✅ conforme · ⚠️ partiel / à nuancer · ❌ écart à traiter.
Pour chaque écart : **P** = corriger le projet, **D** = corriger le document, **P/D** = au choix.

---

## 1. Fiche synthétique du sujet

| Élément | Verdict | Constat |
|---|---|---|
| Thème (pipeline Big Data, temps réel, online learning, détection adaptative du drift) | ✅ | Tous les blocs existent : Kafka + Spark Structured Streaming + Continual-DistilBERT + tri-détecteur ADWIN/KSWIN/PageHinkley + dashboards. |
| Problématique (pipeline open source, économique, reproductible, sans cloud coûteux) | ✅ | Stack 100 % open source, `docker compose up -d`, tourne sur une machine 14 Go de RAM. |
| Objectif général | ✅ | Couvert par l'architecture livrée. |
| **H1** — le drift permet d'identifier statistiquement, avec un délai court, l'émergence d'une nouvelle forme de désinformation | ⚠️ **P/D** | Le mécanisme est implémenté et déclenchable (`scripts/inject_drift_simulation.py`, scénarios A–D). Mais le **tableau de mesure du délai de détection** (12 mesures = 3 algos × 4 scénarios, cf. Tableau 4.3 du mémoire) **n'a pas été régénéré après l'incident** — aucun script ni sortie dans `reports/`. À re-produire, ou à marquer explicitement « non re-mesuré lors de la reprise » dans le mémoire. |
| **H2** — DistilBERT fine-tuné surpasse les approches classiques (TF-IDF, LSTM) | ❌ **P/D** | **Aucune baseline n'a été ré-entraînée.** `scripts/train_model.py` n'entraîne que le DistilBERT. Les valeurs des baselines dans le Tableau 4.1 du mémoire sont issues d'avant l'incident et non reproductibles en l'état. |
| **H3** — le réentraînement adaptatif déclenché par le drift apporte un gain mesurable vs modèle statique (« dérive = opportunité ») | ❌ **P/D** | **Contribution scientifique principale, non re-démontrée après l'incident.** Le mémoire (§4.1.3, §4.4) annonce « +21,8 points de F1 à 3 000 messages post-dérive » — ce chiffre ne provient d'aucun run présent dans le dépôt. Il faut soit produire l'expérience (comparaison Continual vs statique figé sur le scénario B), soit requalifier ce résultat comme « attendu / à valider » dans le mémoire. |

---

## 2. Justification du choix

| Élément | Verdict | Constat |
|---|---|---|
| Justification scientifique (fusion ingénierie de flux + NLP + online learning + éval. statistique de dérive) | ✅ | Cohérent avec l'implémentation. |
| Justification sociétale / opérationnelle (Afrique de l'Ouest, fact-checking, observatoire régional) | ✅ | Enrichissement MasakhaNEWS + sources RSS africaines (Africa News, RFI, Jeune Afrique) présents. |

Aucun écart.

---

## 3. Données et périmètre exploités

### 3.1 Corpus d'entraînement hors ligne

| Affirmation (proposition / mémoire) | Réel (`data/processed/preprocessing_report.json`) | Verdict |
|---|---|---|
| « ~153 064 exemples uniques » | **77 768** après déduplication (**174 059** avant, 96 291 doublons retirés) | ❌ **D** |
| « quatre datasets : ISOT (44 898), WELFake (72 134), FakeNewsNet (23 196), LIAR (12 836) » | Après dédup : **WELFake 44 609 · FakeNewsNet 13 350 · LIAR 9 247**. **ISOT = 0** (ses articles Reuters + sites fake sont déjà tous dans WELFake — recoupement connu). | ❌ **D** — ISOT ne doit plus être présenté comme une source distincte contribuant au corpus final ; le README le documente déjà correctement. |
| « sous-corpus africain MasakhaNEWS, 11 langues » | **12 codes langue** présents (hau, swa, yor, sna, ibo, amh, orm, run, pcm, som, tir, lin) ≈ 10 466 ex. + Google News Africa RSS 96 ex. | ⚠️ **D** — « 11 » → « 12 » (langue manquante dans l'énumération : kirundi/`run`). |
| Split 60/20/20 | **46 660 / 15 554 / 15 554** ✅ | ✅ |
| Corpus équilibré 50/50 fake/réel | ✅ (sous-échantillonnage dans `preprocess_data.py`) | ✅ |

> ⚠️ **Incohérence interne du mémoire v7** : le **Résumé** dit « ~77 768 exemples », l'**Abstract** dit « ~153,064 examples », la section **3.1.1** contient les deux (« Deux datasets » / « quatre datasets » / « ~153 000 » / « ~77 768 »). **À harmoniser sur 77 768 (après dédup) / 174 059 (avant).**

### 3.2 Flux de données en temps réel

| Affirmation | Réel (`producer/src/`) | Verdict |
|---|---|---|
| Flux RSS « 60 sources (40 mainstream + 20 douteuses identifiées par fact-checkers : AFP Factuel, Africa Check, Snopes) » | **12 sources**, toutes catégorisées `reliable` ; `RSS_SOURCES['suspicious'] = []` (vide). Aucune source douteuse branchée. | ❌ **P/D** — soit ajouter des flux de sources douteuses (fichier externe prévu mais absent), soit ramener le chiffre à « 12 sources fiables » dans le mémoire et documenter l'absence de sources douteuses comme limite. |
| API GDELT GKG (quasi temps réel, 15 min) | ✅ Implémentée (`producer/src/gdelt_client.py`, thème `FAKE_NEWS,DISINFORMATION`, intervalle `GDELT_QUERY_INTERVAL_SEC=900`). | ✅ |
| API X (ex-Twitter) Academic — niveau gratuit, 500 000 tweets/mois | **Non implémentée.** Aucune référence à Twitter/X/tweepy dans le code. De plus le tier « Academic Research » gratuit de l'API X **n'existe plus depuis 2023**. | ❌ **P/D** — à retirer du périmètre du mémoire (ou à documenter comme « prévu, non réalisé : accès API X fermé »). |
| Micro-batch Spark « 2 secondes » | `SPARK_MICRO_BATCH_INTERVAL=5 seconds` (`.env.example`, lu par `spark_streaming.py`). | ⚠️ **D** — « 2 s » → « 5 s » (ou réduire la valeur dans `.env` si 2 s tient sur la machine). Apparaît dans le Résumé, l'Abstract et §3.3.3. |

### 3.3 Périmètre analytique

| Élément | Verdict | Constat |
|---|---|---|
| Classification binaire vrai/fake | ✅ | `nlp_classifier.py`, seuil `p_fake ≥ 0.75`. |
| Détection statistique de rupture sur le flux de scores de confiance | ✅ | `drift_monitor.py` : ADWIN/KSWIN/PageHinkley sur `p_fake`. |
| Suivi F1 / latence / débit / délai de détection via tableau de bord | ⚠️ **P/D** | F1 « prequential » temps réel **non affiché** (pas de vérité terrain continue sur le flux de prod — limite légitime, déjà notée dans le dashboard Grafana P7). Latence non journalisée par article. Débit / lag : non exposés. |

---

## 4. Approche méthodologique

### 4.1 Data Engineering

| Élément | Verdict | Constat |
|---|---|---|
| Ingestion Kafka multi-topics | ⚠️ **D** | Topics réels : **`raw-news-stream`** et **`drift-alerts`** uniquement. Le mémoire (§3.3.2, diagramme) mentionne aussi `classified-news` (6 partitions) — ce topic **n'existe plus** (Spark écrit directement dans Mongo/ES). À retirer du diagramme. |
| Spark Structured Streaming en micro-batches | ✅ (voir §3.2 pour la valeur 2 s vs 5 s) | |
| Prétraitement / normalisation multilingue | ✅ | `preprocess_data.py`, `clean_text()`. |
| Historisation MongoDB + Elasticsearch | ✅ | Index ES `articles` + `drift-events` ; collections Mongo `articles` + `drift_events`. |
| Checkpointing du lag | ⚠️ | Checkpoint Spark configuré (`SPARK_CHECKPOINT_DIR`), mais pas de métrique de lag exposée. |
| Conteneurisation complète Docker Compose | ✅ | 13 services, `docker compose config` valide, limites mémoire sur tous les services. |

### 4.2 IA / Data Science

| Élément | Verdict | Constat |
|---|---|---|
| Fine-tuning DistilBERT multilingue | ✅ | `distilbert-base-multilingual-cased`, 10 époques, meilleur checkpoint époque 3, early stopping patience 2. |
| Online learning par mise à jour de gradient + reservoir sampling | ✅ | `online_trainer.py`, `nlp_classifier.py` (reservoir 5 000, 1 batch/5, sync ONNX/100). |
| Tri-détecteur ADWIN + KSWIN + PageHinkley → score composite pondéré | ✅ | Poids 0,45 / 0,35 / 0,20 ; seuil d'alerte **0,4**, seuil de confirmation **0,8**. ⚠️ Le mémoire §3.4.2 / page Drift annoncent parfois « seuil de détection 0,5 » — la valeur réelle est **0,4** (`DRIFT_COMPOSITE_THRESHOLD`). **D**. |
| Export + quantification ONNX INT8 | ✅ | `scripts/export_onnx.py`. Latence mesurée **19,16 ms/article** (et non 5–6 ms — README et mémoire v7 le disent déjà, à propager partout). |

### 4.3 Aide à la décision et Monitoring

| Élément | Verdict | Constat |
|---|---|---|
| Tableau de bord Grafana temps réel | ✅ | `pipeline-desinformation.json` auto-provisionné, 7 panneaux P1–P7, refresh 10 s. |
| Tableau de bord Streamlit | ✅ | `app.py` — 7 pages (Tableau de bord, Articles temps réel, Recherche & Analyse, Drift & Apprentissage, Alertes, Infrastructure, À propos). |
| Système d'alertes | ⚠️ **P/D** | Alertes calculées et affichées dans la page **Alertes** de Streamlit ; évènements de drift persistés MongoDB + ES + topic Kafka `drift-alerts`. **Mais** : aucune **règle d'alerte Grafana provisionnée** (le mémoire RA 5.1 annonce « 3 règles d'alertes configurées, testées, avec captures »). À ajouter dans `config/grafana/provisioning/alerting/`, ou à requalifier dans le mémoire. |

---

## 5. Résultats attendus (mémoire §1.3.2, RA1–RA6)

| # | Livrable attendu | Verdict | Constat |
|---|---|---|---|
| **RA1** | Architecture Big Data opérationnelle et validée | ⚠️ **P** | YAML/config validés (`docker compose config`). **Pas de smoke-test end-to-end des 13 services documenté** après la reprise (checklist + captures manquantes). |
| **RA2** | Modèle NLP haute performance sur corpus massif | ✅ (chiffres) / ⚠️ (formulation) | F1-macro **0,9370**, AUC **0,9899**, AP **0,9900**, précision **0,9387**, rappel **0,9351** — vérifiés (`reports/metrics_summary.json`, `RAPPORT_ANALYSE.md`, 8 figures). Mémoire déjà aligné sur ces valeurs, **mais** contradiction interne « 15 554 » vs « 116 917 exemples de test » dans le §4.1.1 (le bon chiffre est **15 554**). |
| **RA3** | Tri-détecteur calibré **et comparé** (12 mesures : 3 algos × 4 scénarios) | ❌ **P/D** | Détecteur implémenté et calibré (seuils, poids, rémanence). **Tableau comparatif des 12 mesures non régénéré** après l'incident. |
| **RA 3.4** | Infrastructure d'alertes 4 canaux (Mongo + ES + Grafana P4 + Kafka `drift-alerts`) | ⚠️ **P** | 3 canaux OK (Mongo, ES, Kafka). Canal Grafana : panneau P4 présent, mais **règles d'alerte** non provisionnées. |
| **RA4** | Online Learning dynamique — **contribution principale** (« dérive = opportunité », gain ≥ +0,9 pt) | ❌ **P/D** | **Non re-démontré.** Voir H3 (§1). C'est le point le plus sensible pour la soutenance : le mémoire affirme un gain de +21,8 pts non étayé par le dépôt actuel. |
| **RA5** | Dashboard Grafana 7 panneaux + 3 règles d'alertes + Dashboard Streamlit | ⚠️ **P/D** | Grafana 7 panneaux ✅ ; **3 règles d'alertes ❌** ; Streamlit ✅ (le mémoire dit « 5 pages », le dashboard en a **7** — à mettre à jour). Provisioning auto : à re-tester après `docker compose down -v && up -d`. |
| **RA6** | Pipeline déployable < 2 h, scénario soutenance end-to-end, métriques opérationnelles (P95, throughput, lag, uptime), dépôt public MIT, **3 notebooks** | ⚠️/❌ **P/D** | Dépôt public ✅ (GitHub). Licence MIT : **à vérifier** (pas de `LICENSE` constaté). **Tableau 4.3 métriques opérationnelles (P95, throughput, lag, uptime 2 h) non renseigné.** **3 notebooks Jupyter : absents** (seul un stub `notebooks/README`). Scénario soutenance end-to-end non rejoué/documenté. |

---

## 6. Contributions scientifiques et institutionnelles

| Contribution annoncée | Verdict | Constat |
|---|---|---|
| Application intégrée Big Data + IA à la désinformation en flux | ✅ | Livrée. |
| Comparaison rigoureuse de 3 algos de détection de drift | ⚠️ **P/D** | Les 3 algos sont combinés et fonctionnent ; la **comparaison chiffrée** (délai, TPR, FPR par scénario) reste à régénérer. |
| Démonstration « une dérive peut être transformée en gain de performance » | ❌ **P/D** | **Non étayée par le dépôt actuel** (voir H3 / RA4). Contribution *principale* → priorité n°1 avant soutenance. |
| Mémoire de référence + dépôt pédagogique réutilisable pour UCAO-UUT | ✅ | Code documenté, Docker reproductible, README détaillé, rapport d'analyse + figures. |
| Prototype open source pour le fact-checking ouest-africain | ✅ | Publié, enrichissement africain présent (avec la limite ~77 fake africaines authentiques, déjà documentée). |

---

## Synthèse — ce qui bloque une conformité totale

### Écarts « projet » (implémentation à compléter si on veut cocher la case telle qu'écrite)
1. **Expérience « dérive = opportunité » (RA4 / H3 / contribution principale)** — comparaison Continual-DistilBERT vs DistilBERT statique figé sur le scénario B, avec courbe de F1 avant/après dérive. **Priorité 1.**
2. **Tableau comparatif du tri-détecteur (RA3)** — 3 algos × 4 scénarios : délai de détection, TPR, FPR.
3. **Baselines statiques (H2)** — au minimum TF-IDF + régression logistique et un LSTM, sur le même split.
4. **Règles d'alerte Grafana (RA3.4 / RA5)** — `config/grafana/provisioning/alerting/`.
5. **Métriques opérationnelles (RA6.3)** — latence médiane / P95, throughput, lag max, uptime sur 2 h.
6. **3 notebooks Jupyter (RA6.4)** — perdus à l'incident, à reconstruire ou à retirer du mémoire.
7. **Smoke-test end-to-end des 13 services + checklist (RA1 / RA6.1)**.
8. **`LICENSE` MIT** à ajouter à la racine (annoncé RA6.4).
9. *(optionnel)* Flux RSS de sources douteuses (périmètre §3.1.2).

### Écarts « document » (chiffres/affirmations du mémoire à corriger — le projet a raison)
| Endroit du mémoire | Écrit | Corriger en |
|---|---|---|
| Résumé / Abstract / §3.1.1 | corpus « ~153 064 » / « 4 datasets dont ISOT » | **77 768** après dédup (174 059 avant) ; ISOT **absorbé** par WELFake, ne plus le compter comme source |
| Abstract vs Résumé | 153 064 (EN) ≠ 77 768 (FR) | valeur unique **77 768** |
| Résumé / Abstract / §3.3.3 | micro-batch « 2 secondes » | **5 secondes** |
| §3.1.1 | MasakhaNEWS « 11 langues » | **12** (ajouter le kirundi) |
| §3.1.2 | « 60 sources RSS (40 + 20 douteuses) » | **12 sources fiables**, 0 douteuse (ou implémenter) |
| §3.1.2 | « API X Academic — niveau gratuit » | **retirer** (non implémenté, tier fermé depuis 2023) |
| §3.3.2 + diagramme | topic Kafka `classified-news` | **retirer** (n'existe pas : `raw-news-stream` + `drift-alerts`) |
| §3.4.2 / dashboards | seuil de détection du drift « 0,5 » | **0,4** (`DRIFT_COMPOSITE_THRESHOLD`) |
| §4.1.1 | « jeu de test complet 116 917 exemples » | **15 554** |
| §1.3.2 RA5.1 | « Streamlit 5 pages » | **7 pages** |
| Partout | latence ONNX « 5–6 ms » | **19,16 ms/article** (mesurée) — déjà partiellement fait |
| §4.1.1 / §4.1.3 / §4.4 | Tableaux 4.1, 4.3, 4.4 + « +21,8 pts », « RoBERTa F1=0,921 », baselines | marquer **« résultats d'avant l'incident du 26/08/2026, non re-vérifiés »** tant que les expériences ne sont pas rejouées |

---

_Références : `data/processed/preprocessing_report.json`, `reports/metrics_summary.json`,
`reports/RAPPORT_ANALYSE.md`, `producer/src/rss_sources.py`, `producer/src/gdelt_client.py`,
`spark-app/src/drift_monitor.py`, `spark-app/src/spark_streaming.py`, `.env.example`,
`api/src/routers/`, `config/grafana/`._
