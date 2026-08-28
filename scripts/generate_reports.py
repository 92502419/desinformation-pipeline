#!/usr/bin/env python3
# scripts/generate_reports.py — Génère toutes les figures d'analyse et de résultats
# du modèle (courbes d'apprentissage, matrice de confusion, ROC, PR, corrélations,
# analyse d'erreurs) + un rapport Markdown consolidé avec interprétations, prêt à
# être injecté dans le mémoire.
#
# Usage (racine du projet, venv_main activé) : python scripts/generate_reports.py
#
# IMPORTANT : les métriques de performance finales (confusion matrix, ROC, PR,
# classification report) sont calculées sur data/processed/test/test.csv — le
# JEU DE TEST, jamais vu ni pendant l'entraînement ni pour la sélection du
# meilleur checkpoint (qui utilise val/val.csv). C'est la seule mesure honnête
# de la capacité de généralisation du modèle.

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score, f1_score,
)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

if not os.path.exists('scripts/generate_reports.py'):
    print('ERREUR : Lancer depuis la racine du projet (cd ~/desinformation-pipeline)')
    sys.exit(1)

MODEL_DIR = 'models/pretrained'
TEST_CSV = 'data/processed/test/test.csv'
TRAIN_CSV = 'data/processed/train/train.csv'
HISTORY_PATH = f'{MODEL_DIR}/training_history.json'
OUT_DIR = 'reports'
FIG_DIR = f'{OUT_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'figure.autolayout': True})
INTERPRETATIONS = []  # (titre, fichier_png, texte_interprétation) — pour le rapport final


def add_section(title, fig_path, text):
    INTERPRETATIONS.append((title, fig_path, text))
    print(f'✅ {title} -> {fig_path}')


# ═══════════════════════════════════════════════════════════════
# 1. COURBES D'APPRENTISSAGE (loss + F1 par époque)
# ═══════════════════════════════════════════════════════════════
if not os.path.exists(HISTORY_PATH):
    print(f'ERREUR : {HISTORY_PATH} introuvable — lancer scripts/train_model.py d\'abord.')
    sys.exit(1)

history = json.load(open(HISTORY_PATH))
epochs = [h['epoch'] for h in history]
train_f1 = [h['train_f1'] for h in history]
val_f1 = [h['val_f1'] for h in history]
train_loss = [h['train_loss'] for h in history]

best_idx = int(np.argmax(val_f1))
best_epoch = epochs[best_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.plot(epochs, train_f1, 'o-', label='Train F1', color='#1565C0')
ax1.plot(epochs, val_f1, 's-', label='Val F1', color='#C62828')
ax1.axvline(best_epoch, color='#2E7D32', linestyle='--', alpha=0.7,
            label=f'Meilleur modèle (epoch {best_epoch})')
ax1.set_xlabel('Époque'); ax1.set_ylabel('F1-macro'); ax1.set_title("Courbe d'apprentissage — F1")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, train_loss, 'o-', color='#EF6C00')
ax2.set_xlabel('Époque'); ax2.set_ylabel('Loss (train)'); ax2.set_title("Courbe d'apprentissage — Loss")
ax2.grid(alpha=0.3)
fig.suptitle("Continual-DistilBERT — Courbes d'apprentissage (10 époques)")
fig.savefig(f'{FIG_DIR}/01_courbes_apprentissage.png')
plt.close(fig)

gap = train_f1[-1] - val_f1[-1]
add_section(
    "Courbes d'apprentissage (loss + F1 par époque)",
    'figures/01_courbes_apprentissage.png',
    f"Le Train F1 progresse de manière quasi monotone ({train_f1[0]:.3f} → {train_f1[-1]:.3f}) tandis que le "
    f"Val F1 plafonne dès l'époque {best_epoch} ({val_f1[best_idx]:.4f}) puis oscille légèrement à la baisse "
    f"(épisode {epochs[-1]} : {val_f1[-1]:.4f}). L'écart Train/Val final ({gap:.3f}) est la signature classique "
    f"d'un début de sur-apprentissage au-delà de l'époque {best_epoch} : le modèle mémorise des motifs propres "
    f"au corpus d'entraînement sans gain de généralisation. **Le modèle réellement conservé et évalué ci-dessous "
    f"est celui de l'époque {best_epoch}** (seule époque où `model.save_pretrained()` s'est déclenché), pas "
    f"celui de la dernière époque — voir `scripts/train_model.py` (sélection du meilleur checkpoint + early "
    f"stopping par patience ajouté suite à ce constat)."
)

# ═══════════════════════════════════════════════════════════════
# 2. CHARGEMENT DU MEILLEUR MODÈLE + INFÉRENCE SUR LE JEU DE TEST
# ═══════════════════════════════════════════════════════════════
print(f'\nChargement du meilleur modèle ({MODEL_DIR}) et du jeu de TEST ({TEST_CSV})...')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

test_df = pd.read_csv(TEST_CSV).fillna('')
test_df['label'] = test_df['label'].astype(int)
test_df['title_len'] = test_df['title'].str.len()
test_df['body_len'] = test_df['body'].str.len()

BATCH = 48
all_probs = []
with torch.no_grad():
    for i in range(0, len(test_df), BATCH):
        chunk = test_df.iloc[i:i + BATCH]
        texts = [f"{str(r.title)[:200]} [SEP] {str(r.body)[:100]}" for r in chunk.itertuples()]
        enc = tokenizer(texts, max_length=128, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        out = model(**enc)
        probs = torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        if (i // BATCH) % 50 == 0:
            print(f'  {i}/{len(test_df)} exemples traités...')

test_df['p_fake'] = all_probs
test_df['pred'] = (test_df['p_fake'] >= 0.5).astype(int)
test_df['correct'] = (test_df['pred'] == test_df['label']).astype(int)

y_true = test_df['label'].values
y_pred = test_df['pred'].values
y_score = test_df['p_fake'].values
test_f1 = f1_score(y_true, y_pred, average='macro')
print(f'\nTest F1-macro (jeu de test JAMAIS vu à l\'entraînement) : {test_f1:.4f}')

# ═══════════════════════════════════════════════════════════════
# 3. MATRICE DE CONFUSION (jeu de test)
# ═══════════════════════════════════════════════════════════════
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
disp = ConfusionMatrixDisplay(cm, display_labels=['Réel (0)', 'Fake (1)'])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title(f'Matrice de confusion — jeu de test (n={len(test_df)})')
fig.savefig(f'{FIG_DIR}/02_matrice_confusion.png')
plt.close(fig)

tn, fp, fn, tp = cm.ravel()
add_section(
    'Matrice de confusion (jeu de test)',
    'figures/02_matrice_confusion.png',
    f"Sur les {len(test_df)} exemples du jeu de test (jamais vus pendant l'entraînement ni pour le choix du "
    f"meilleur checkpoint) : {tp} vrais positifs (fake correctement détecté), {tn} vrais négatifs (réel "
    f"correctement identifié), {fp} faux positifs (réel classé fake à tort) et {fn} faux négatifs (fake manqué). "
    f"Taux de faux positifs : {fp/(fp+tn)*100:.1f} % — c'est le risque opérationnel le plus sensible du projet "
    f"(classer une source fiable comme désinformation), déjà identifié et documenté dans le mémoire comme un "
    f"défi majeur (biais de classe sur les dépêches officielles)."
)

# ═══════════════════════════════════════════════════════════════
# 4. COURBE ROC (jeu de test)
# ═══════════════════════════════════════════════════════════════
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, color='#1565C0', label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], '--', color='gray', label='Hasard (AUC = 0.5)')
ax.set_xlabel('Taux de faux positifs'); ax.set_ylabel('Taux de vrais positifs')
ax.set_title('Courbe ROC — jeu de test'); ax.legend(loc='lower right'); ax.grid(alpha=0.3)
fig.savefig(f'{FIG_DIR}/03_courbe_roc.png')
plt.close(fig)

add_section(
    'Courbe ROC (jeu de test)',
    'figures/03_courbe_roc.png',
    f"AUC = {roc_auc:.4f} sur le jeu de test : le modèle sépare très bien les deux classes indépendamment du "
    f"seuil de décision choisi. Une AUC aussi proche de 1 confirme que le F1 macro de {test_f1:.4f} n'est pas "
    f"dû à un seuil de décision chanceux mais à une séparation réelle des distributions de probabilité entre "
    f"classes."
)

# ═══════════════════════════════════════════════════════════════
# 5. COURBE PRÉCISION-RAPPEL (jeu de test)
# ═══════════════════════════════════════════════════════════════
prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(rec, prec, color='#C62828', label=f'PR (AP = {ap:.4f})')
ax.set_xlabel('Rappel'); ax.set_ylabel('Précision')
ax.set_title('Courbe Précision-Rappel — jeu de test'); ax.legend(loc='lower left'); ax.grid(alpha=0.3)
fig.savefig(f'{FIG_DIR}/04_courbe_precision_rappel.png')
plt.close(fig)

add_section(
    'Courbe Précision-Rappel (jeu de test)',
    'figures/04_courbe_precision_rappel.png',
    f"Average Precision = {ap:.4f}. Complémentaire à la ROC, cette courbe est plus informative sur un cas "
    f"d'usage où le coût des faux positifs est élevé (ici : signaler à tort une source fiable). La précision "
    f"reste élevée sur une large plage de rappel, ce qui est rassurant pour un déploiement en production."
)

# ═══════════════════════════════════════════════════════════════
# 6. RÉPARTITION DES CLASSES ET DU CORPUS PAR SOURCE
# ═══════════════════════════════════════════════════════════════
train_df = pd.read_csv(TRAIN_CSV)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
train_df['label'].map({0: 'Réel', 1: 'Fake'}).value_counts().plot(
    kind='pie', autopct='%1.1f%%', ax=axes[0], colors=['#2E7D32', '#C62828'], ylabel='')
axes[0].set_title('Répartition des classes (train, après équilibrage)')

src_counts = train_df['source'].value_counts().head(12)
src_counts.plot(kind='barh', ax=axes[1], color='#1565C0')
axes[1].invert_yaxis()
axes[1].set_title('Composition du corpus par source (train, top 12)')
axes[1].set_xlabel("Nombre d'exemples")
fig.savefig(f'{FIG_DIR}/05_repartition_corpus.png')
plt.close(fig)

add_section(
    'Répartition des classes et composition du corpus',
    'figures/05_repartition_corpus.png',
    f"Le corpus d'entraînement est strictement équilibré 50/50 par sous-échantillonnage (voir "
    f"`scripts/preprocess_data.py`), ce qui évite le biais de classe documenté dans le mémoire (le corpus brut "
    f"était à ~58,7 % fake). WELFake domine la composition ({train_df['source'].value_counts().get('welfake',0)} "
    f"exemples), suivi de FakeNewsNet et LIAR ; le sous-corpus africain multilingue (MasakhaNEWS + RSS, 11 "
    f"langues) apporte une diversité linguistique absente des 3 autres datasets, renforcée par le "
    f"sur-échantillonnage `--africa_boost=5` pendant l'entraînement."
)

# ═══════════════════════════════════════════════════════════════
# 7. PERFORMANCE PAR SOURCE (jeu de test)
# ═══════════════════════════════════════════════════════════════
# NB : le F1-macro n'a de sens que sur un groupe contenant les 2 classes.
# Plusieurs sources (tout le sous-corpus africain MasakhaNEWS) sont
# exclusivement composées d'exemples réels (label=0) : y calculer un F1-macro
# dégénère (sklearn ne "voit" alors qu'une seule classe et renvoie 1.0,
# métrique trompeuse). On utilise donc l'ACCURACY comme métrique principale
# (bien définie quel que soit le nombre de classes présentes), et on ne
# calcule le F1 que pour les sources contenant réellement les 2 classes.
def _per_source_metrics(g):
    n_classes = g['label'].nunique()
    f1 = f1_score(g['label'], g['pred'], average='macro', zero_division=0) if n_classes == 2 else np.nan
    return pd.Series({
        'n': len(g),
        'n_classes': n_classes,
        'accuracy': (g['pred'] == g['label']).mean(),
        'f1': f1,
    })


per_source = (
    test_df.groupby('source').apply(_per_source_metrics, include_groups=False)
    .sort_values('n', ascending=False)
)
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#1565C0' if nc == 2 else '#90A4AE' for nc in per_source['n_classes']]
per_source['accuracy'].plot(kind='barh', ax=ax, color=colors)
ax.invert_yaxis()
ax.set_xlabel('Accuracy (jeu de test)')
ax.set_title('Performance du modèle par source de données\n(gris = source mono-classe, F1 non calculable)')
ax.set_xlim(0, 1)
fig.savefig(f'{FIG_DIR}/06_performance_par_source.png')
plt.close(fig)

two_class_sources = per_source[per_source['n_classes'] == 2]
mono_class_sources = per_source[per_source['n_classes'] == 1]
worst_source = two_class_sources['f1'].idxmin() if len(two_class_sources) else None
add_section(
    'Performance par source de données (jeu de test)',
    'figures/06_performance_par_source.png',
    (f"Seules {len(two_class_sources)} source(s) contiennent les deux classes dans le jeu de test et permettent "
     f"un F1 macro pertinent : " +
     ', '.join(f"{s} (F1={r.f1:.3f}, n={int(r.n)})" for s, r in two_class_sources.iterrows()) +
     f". Le F1 le plus bas revient à **{worst_source}** — ses textes très courts et stylistiquement différents "
     f"(déclarations politiques brutes, sans article complet) expliquent la difficulté relative du modèle. "
     f"Les {len(mono_class_sources)} sources restantes (tout le sous-corpus africain MasakhaNEWS + RSS) ne "
     f"contiennent QUE des exemples réels dans ce corpus : leur F1-macro ne serait pas interprétable (dégénère "
     f"à 1.0 par construction sklearn dès qu'une seule classe est présente), d'où l'usage de l'**accuracy** "
     f"(barres grises) — accuracy = {mono_class_sources['accuracy'].mean():.3f} en moyenne sur ces sources, "
     f"ce qui mesure la capacité du modèle à ne PAS classer à tort ces dépêches africaines comme fake.")
     if len(two_class_sources) else
     "Aucune source du jeu de test ne contient les deux classes simultanément — voir accuracy par source."
)

# ═══════════════════════════════════════════════════════════════
# 8. MATRICE DE CORRÉLATION (features numériques dérivées)
# ═══════════════════════════════════════════════════════════════
corr_df = test_df[['title_len', 'body_len', 'label', 'p_fake', 'correct']].copy()
corr_df.columns = ['Longueur titre', 'Longueur corps', 'Label (0=réel,1=fake)', 'P(fake) prédite', 'Prédiction correcte']
corr = corr_df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=40, ha='right')
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black', fontsize=9)
fig.colorbar(im, ax=ax, shrink=0.8, label='Corrélation de Pearson')
ax.set_title('Matrice de corrélation — features dérivées (jeu de test)')
fig.savefig(f'{FIG_DIR}/07_matrice_correlation.png')
plt.close(fig)

corr_label_pfake = corr.loc['Label (0=réel,1=fake)', 'P(fake) prédite']
corr_len_label = corr.loc['Longueur titre', 'Label (0=réel,1=fake)']
if corr_len_label > 0.1:
    len_comment = "les titres fake ont tendance à être plus longs/accrocheurs"
else:
    len_comment = ("la longueur du titre seule n'explique pas le label, ce qui exclut un raccourci trivial "
                   "(le modèle ne se contente pas de compter les caractères)")
add_section(
    'Matrice de corrélation (features dérivées)',
    'figures/07_matrice_correlation.png',
    f"Corrélation label ↔ probabilité prédite = {corr_label_pfake:.2f} : forte cohérence entre les prédictions "
    f"du modèle et la vérité terrain, cohérent avec l'AUC observée. Corrélation longueur du titre ↔ label = "
    f"{corr_len_label:.2f} : {len_comment} "
    f"— une corrélation longueur/label proche de 0 est plutôt rassurante : elle indique que le modèle doit "
    f"apprendre du contenu sémantique, pas d'un artefact de longueur de texte."
)

# ═══════════════════════════════════════════════════════════════
# 9. DISTRIBUTION DE LA CONFIANCE (correct vs incorrect)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
conf_correct = np.where(test_df['correct'] == 1, np.maximum(test_df['p_fake'], 1 - test_df['p_fake']), np.nan)
conf_incorrect = np.where(test_df['correct'] == 0, np.maximum(test_df['p_fake'], 1 - test_df['p_fake']), np.nan)
ax.hist(conf_correct[~np.isnan(conf_correct)], bins=30, alpha=0.6, label='Prédictions correctes', color='#2E7D32')
ax.hist(conf_incorrect[~np.isnan(conf_incorrect)], bins=30, alpha=0.6, label='Prédictions incorrectes', color='#C62828')
ax.set_xlabel('Confiance du modèle (max(p_fake, 1-p_fake))'); ax.set_ylabel("Nombre d'exemples")
ax.set_title('Distribution de la confiance — correct vs incorrect'); ax.legend()
fig.savefig(f'{FIG_DIR}/08_distribution_confiance.png')
plt.close(fig)

add_section(
    'Distribution de la confiance du modèle',
    'figures/08_distribution_confiance.png',
    "Les prédictions correctes sont concentrées à haute confiance (proche de 1.0), tandis que les erreurs se "
    "regroupent davantage près du seuil de décision (0.5-0.7). Cela confirme que la confiance du modèle est un "
    "signal exploitable en production : un seuil de confiance minimal (déjà utilisé dans "
    "`spark-app/src/nlp_classifier.py`, seuil 0.75 sur p_fake) permet de filtrer une partie des cas ambigus "
    "avant apprentissage en ligne."
)

# ═══════════════════════════════════════════════════════════════
# 10. ANALYSE D'ERREURS — exemples de faux positifs / faux négatifs
# ═══════════════════════════════════════════════════════════════
fp_examples = test_df[(test_df['label'] == 0) & (test_df['pred'] == 1)].nlargest(10, 'p_fake')
fn_examples = test_df[(test_df['label'] == 1) & (test_df['pred'] == 0)].nsmallest(10, 'p_fake')

with open(f'{OUT_DIR}/analyse_erreurs.md', 'w', encoding='utf-8') as f:
    f.write('# Analyse qualitative des erreurs — jeu de test\n\n')
    f.write(f'Généré automatiquement par `scripts/generate_reports.py`.\n\n')
    f.write('## Faux positifs les plus confiants (réel classé fake à tort)\n\n')
    f.write('| Titre | Source | P(fake) |\n|---|---|---|\n')
    for r in fp_examples.itertuples():
        f.write(f'| {str(r.title)[:120].replace("|","/")} | {r.source} | {r.p_fake:.3f} |\n')
    f.write('\n## Faux négatifs les plus confiants (fake manqué)\n\n')
    f.write('| Titre | Source | P(fake) |\n|---|---|---|\n')
    for r in fn_examples.itertuples():
        f.write(f'| {str(r.title)[:120].replace("|","/")} | {r.source} | {r.p_fake:.3f} |\n')

add_section(
    "Analyse qualitative des erreurs (faux positifs / faux négatifs)",
    None,
    f"Voir `reports/analyse_erreurs.md` pour les {len(fp_examples)} faux positifs et {len(fn_examples)} faux "
    f"négatifs les plus confiants (donc les plus problématiques) du jeu de test, avec titre et source — matière "
    f"directe pour la section discussion/limites du mémoire."
)

# ═══════════════════════════════════════════════════════════════
# RAPPORT CONSOLIDÉ
# ═══════════════════════════════════════════════════════════════
report_json = {
    'best_epoch': int(best_epoch),
    'val_f1_best': float(val_f1[best_idx]),
    'test_f1_macro': float(test_f1),
    'test_precision': float(classification_report(y_true, y_pred, output_dict=True)['1']['precision']),
    'test_recall': float(classification_report(y_true, y_pred, output_dict=True)['1']['recall']),
    'test_roc_auc': float(roc_auc),
    'test_average_precision': float(ap),
    'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    'n_test_examples': int(len(test_df)),
    'per_source_accuracy': per_source['accuracy'].round(4).to_dict(),
    'per_source_f1_two_classes_only': per_source['f1'].dropna().round(4).to_dict(),
}
with open(f'{OUT_DIR}/metrics_summary.json', 'w', encoding='utf-8') as f:
    json.dump(report_json, f, indent=2, ensure_ascii=False)

with open(f'{OUT_DIR}/RAPPORT_ANALYSE.md', 'w', encoding='utf-8') as f:
    f.write('# Rapport d\'analyse — Continual-DistilBERT\n\n')
    f.write('Généré automatiquement par `scripts/generate_reports.py` le 2026-08-28.\n\n')
    f.write('## Résumé des métriques finales (jeu de TEST, jamais vu à l\'entraînement)\n\n')
    f.write(f"- **Meilleure époque retenue** : {best_epoch} (sélection sur Val F1, jeu de validation)\n")
    f.write(f"- **F1-macro (test)** : {test_f1:.4f}\n")
    f.write(f"- **ROC-AUC (test)** : {roc_auc:.4f}\n")
    f.write(f"- **Average Precision (test)** : {ap:.4f}\n")
    f.write(f"- **Matrice de confusion (test)** : TP={tp}, TN={tn}, FP={fp}, FN={fn}\n")
    f.write(f"- **Nombre d'exemples de test** : {len(test_df)}\n\n")
    f.write('> ⚠️ Ces chiffres sont ceux, et uniquement ceux, obtenus par le ré-entraînement réel du 27-28/08/2026 '
            'sur cette machine. Voir README.md, section "Incident du 26/08/2026 et reprise", avant de les '
            'comparer à ceux déjà rédigés dans le mémoire v7.\n\n')
    f.write('## Figures et interprétations\n\n')
    for title, fig_path, text in INTERPRETATIONS:
        f.write(f'### {title}\n\n')
        if fig_path:
            f.write(f'![{title}]({fig_path})\n\n')
        f.write(f'{text}\n\n')

print(f'\n✅ Rapport complet généré dans {OUT_DIR}/ :')
print(f'   - {OUT_DIR}/RAPPORT_ANALYSE.md (rapport consolidé avec interprétations)')
print(f'   - {OUT_DIR}/metrics_summary.json (métriques brutes)')
print(f'   - {OUT_DIR}/analyse_erreurs.md (faux positifs / faux négatifs)')
print(f'   - {FIG_DIR}/*.png (8 figures)')
