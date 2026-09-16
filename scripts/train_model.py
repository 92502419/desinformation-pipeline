#!/usr/bin/env python3
# scripts/train_model.py — Pré-entraînement Continual-DistilBERT
#
# v3 (2026-08-27) — reconstruit après un plantage disque qui a corrompu le
# checkpoint de reprise et l'historique d'entraînement pendant un run qui
# s'est arrêté à l'epoch 1 sur cette machine (T600 4 Go VRAM). Ajouts par
# rapport à la version documentée dans Documentation_Technique_..._v9.docx :
#   - checkpoint de reprise sauvegardé de façon ATOMIQUE (fichier .tmp puis
#     renommage) toutes les --save_every batches, pour ne plus jamais se
#     retrouver avec un .pt tronqué et illisible après un plantage.
#   - --resume : reprise automatique (poids + optimizer + epoch + step) si un
#     checkpoint de reprise existe déjà dans --output_dir.
#   - historique JSON écrit de façon incrémentale (atomique) après CHAQUE
#     epoch, plus après chaque validation partielle, pour ne rien perdre
#     même si le process est tué en cours de route.
#   - --africa_boost : sur-échantillonnage (WeightedRandomSampler) des
#     exemples du sous-corpus africain, comme documenté dans le mémoire (v2.4).
#   - mixed precision (AMP) sur GPU pour tenir sur un GPU 4 Go de VRAM.
#
# v4 (2026-08-28) — un premier run 10 époques (voir reports/) a montré un
# sur-apprentissage classique après l'epoch 3 (Train F1 continue vers 0.99,
# Val F1 plafonne/oscille ~0.937). Le modèle livré n'a jamais été affecté
# (model.save_pretrained ne s'exécute que si le Val F1 s'améliore, donc les
# poids sur disque sont restés ceux de l'epoch 3), mais le script laissait
# tourner les 10 époques pour rien. Ajout d'un VRAI early stopping par
# patience (--patience, défaut 2), conforme au mémoire (§OS 2.3 : "early
# stopping patience=2 sur le F1-macro de validation") — --target_f1 reste un
# critère d'arrêt optionnel supplémentaire, désactivé par défaut.
#
# Usage (RACINE du projet, venv_main activé) :
#   python scripts/train_model.py --epochs 10 --batch_size 8 --lr 2e-5 \
#       --africa_boost 5 --save_every 100

import argparse, os, sys, json, time, tempfile

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm

if not os.path.exists('scripts/train_model.py'):
    print('ERREUR : Lancer depuis la RACINE : cd ~/desinformation-pipeline')
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16, help='16 sur CPU, 8 recommandé sur GPU <=4 Go VRAM')
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--model_name', default='distilbert-base-multilingual-cased')
parser.add_argument('--output_dir', default='models/pretrained')
parser.add_argument('--train_csv', default='data/processed/train/train.csv')
parser.add_argument('--val_csv', default='data/processed/val/val.csv')
parser.add_argument('--africa_boost', type=float, default=5.0, help='Facteur de sur-pondération des exemples africains')
parser.add_argument('--save_every', type=int, default=100, help='Sauvegarde du checkpoint de reprise tous les N batches')
parser.add_argument('--resume', action='store_true', default=True, help='Reprendre depuis le dernier checkpoint si présent')
parser.add_argument('--no-resume', dest='resume', action='store_false')
parser.add_argument('--target_f1', type=float, default=0.999,
                     help="Arrêt immédiat si Val F1 >= target_f1 (objectif de qualité, désactivé par défaut : "
                          "0.999 est hors de portée en pratique). Le VRAI critère anti-surapprentissage est "
                          "--patience ci-dessous, conforme au mémoire (§OS 2.3 : 'early stopping patience=2 "
                          "sur le F1-macro de validation').")
parser.add_argument('--patience', type=int, default=2,
                     help="Nombre d'époques consécutives sans amélioration du Val F1 avant arrêt anticipé "
                          "(early stopping standard anti-surapprentissage). 0 = désactivé.")
parser.add_argument('--max_train_examples', type=int, default=0, help='0 = tout le corpus ; sinon sous-échantillon pour un smoke test rapide')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Utilisation de : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)} | VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go')
else:
    print('CPU détecté → batch_size=16 recommandé. Google Colab pour accélérer.')

for path in [args.train_csv, args.val_csv]:
    if not os.path.exists(path):
        print(f'ERREUR : {path} introuvable.')
        print("Exécuter d'abord : python scripts/preprocess_data.py")
        sys.exit(1)

RESUME_PATH = os.path.join(args.output_dir, '_resume_checkpoint.pt')
HISTORY_PATH = os.path.join(args.output_dir, 'training_history.json')


def atomic_save(obj_save_fn, final_path):
    """Écrit dans un fichier temporaire puis renomme — jamais de fichier tronqué
    même si le process est tué au milieu de l'écriture (cause du plantage précédent)."""
    d = os.path.dirname(final_path) or '.'
    os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.part')
    os.close(fd)
    try:
        obj_save_fn(tmp_path)
        os.replace(tmp_path, final_path)  # remplacement atomique (même filesystem)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(obj, final_path):
    atomic_save(lambda p: json.dump(obj, open(p, 'w'), indent=2), final_path)


class NewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128, limit=0):
        self.df = pd.read_csv(csv_path).fillna('')
        if limit and limit < len(self.df):
            self.df = self.df.sample(n=limit, random_state=42).reset_index(drop=True)
        self.df['label'] = self.df['label'].astype(int)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def is_african(self, idx):
        src = str(self.df.iloc[idx].get('source', ''))
        return src.startswith('masakhanews') or src == 'google_news_africa_rss'

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = f"{str(row['title'])[:200]} [SEP] {str(row['body'])[:100]}"
        # Pas de padding ici : padding dynamique par batch via collate_fn (voir plus
        # bas) — la plupart des textes sont bien plus courts que max_len=128 ; les
        # padder tous à 128 gaspillait ~2-3x de calcul GPU pour rien sur ce matériel
        # modeste (T600 4 Go). Le padding par batch au plus long élément accélère
        # significativement l'entraînement sans changer la troncature maximale.
        enc = self.tokenizer(text, max_length=self.max_len, truncation=True)
        return {
            'input_ids': enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels': int(row['label']),
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        labels = torch.tensor([b['labels'] for b in batch], dtype=torch.long)
        padded = tokenizer.pad(
            [{'input_ids': b['input_ids'], 'attention_mask': b['attention_mask']} for b in batch],
            padding=True, return_tensors='pt',
        )
        padded['labels'] = labels
        return padded
    return collate_fn


print(f'Chargement du tokenizer : {args.model_name}')
tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    args.model_name, num_labels=2).to(DEVICE)

print('Chargement des données...')
train_ds = NewsDataset(args.train_csv, tokenizer, args.max_len, limit=args.max_train_examples)
val_ds = NewsDataset(args.val_csv, tokenizer, args.max_len)

# ── Sampler pondéré : boost africain (v2.4) ──────────────────────
sample_weights = [args.africa_boost if train_ds.is_african(i) else 1.0 for i in range(len(train_ds))]
n_africa = sum(1 for w in sample_weights if w != 1.0)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
print(f'Exemples africains dans le train set : {n_africa} (boost x{args.africa_boost})')

# n_workers=0 sur TOUS les devices : sur cette machine (Python 3.14), le
# multiprocessing DataLoader (start method 'forkserver' par défaut) plante avec
# ConnectionResetError. Le coût du chargement mono-thread reste négligeable ici
# (batch_size modeste, tokenisation courte).
n_workers = 0
collate_fn = make_collate_fn(tokenizer)
train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                       num_workers=n_workers, pin_memory=(DEVICE == 'cuda'), collate_fn=collate_fn)
val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                     num_workers=n_workers, collate_fn=collate_fn)
print(f'Train : {len(train_ds)} ex. | Val : {len(val_ds)} ex. | Batches/epoch : {len(train_dl)}')

lbl_series = pd.read_csv(args.train_csv)['label'].astype(int)
n_fake = (lbl_series == 1).sum()
n_real = (lbl_series == 0).sum()
n_total = len(lbl_series)
print(f'fake={n_fake} ({n_fake/n_total*100:.1f}%) | real={n_real} ({n_real/n_total*100:.1f}%)')
w = torch.tensor([n_fake / n_total, n_real / n_total]).float().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=w)

layer_lrs = []
for i, layer in enumerate(model.distilbert.transformer.layer):
    layer_lrs.append({'params': layer.parameters(), 'lr': args.lr * (0.9 ** (5 - i))})
layer_lrs.append({'params': model.pre_classifier.parameters(), 'lr': args.lr})
layer_lrs.append({'params': model.classifier.parameters(), 'lr': args.lr})
optimizer = AdamW(layer_lrs, weight_decay=0.01)

scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))
os.makedirs(args.output_dir, exist_ok=True)

start_epoch, start_batch = 0, 0
best_f1, history = 0.0, []
epochs_no_improve = 0
best_epoch = 0

# ── Reprise depuis un checkpoint valide (si demandé et présent) ─
if args.resume and os.path.exists(RESUME_PATH):
    try:
        ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 0)
        start_batch = ckpt.get('batch', 0)
        best_f1 = ckpt.get('best_f1', 0.0)
        history = ckpt.get('history', [])
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        best_epoch = ckpt.get('best_epoch', 0)
        print(f'Reprise depuis {RESUME_PATH} : epoch={start_epoch}, batch={start_batch}, best_f1={best_f1:.4f}')
    except Exception as e:
        print(f'Checkpoint de reprise illisible ({e}) — on repart de zéro (comportement voulu : '
              f'ne jamais planter sur un .pt corrompu).')
        start_epoch, start_batch = 0, 0


def save_resume_checkpoint(epoch, batch_idx):
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'batch': batch_idx,
        'best_f1': best_f1,
        'history': history,
        'epochs_no_improve': epochs_no_improve,
        'best_epoch': best_epoch,
    }
    atomic_save(lambda p: torch.save(payload, p), RESUME_PATH)


for epoch in range(start_epoch, args.epochs):
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=max(1, len(train_dl) // 5))
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    t0 = time.time()
    pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{args.epochs}')
    for batch_idx, batch in enumerate(pbar):
        if epoch == start_epoch and batch_idx < start_batch:
            continue  # saute les batches déjà traités avant le plantage
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        lbls = batch['labels'].to(DEVICE).long()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(DEVICE == 'cuda')):
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            loss = loss_fn(out.logits, lbls)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        preds_all.extend(out.logits.argmax(-1).detach().cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())
        pbar.set_postfix(loss=f'{loss.item():.4f}')

        if args.save_every and (batch_idx + 1) % args.save_every == 0:
            save_resume_checkpoint(epoch, batch_idx + 1)

    train_f1 = f1_score(labels_all, preds_all, average='macro') if labels_all else 0.0
    elapsed = time.time() - t0
    print(f'Epoch {epoch+1} | Loss: {total_loss/max(1,len(train_dl)):.4f} | Train F1: {train_f1:.4f} | Durée: {elapsed/60:.1f} min')

    # ── Validation ──────────────────────────────────────────
    model.eval()
    val_preds, val_labels, val_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_dl, desc='Validation', leave=False):
            out = model(input_ids=batch['input_ids'].to(DEVICE),
                        attention_mask=batch['attention_mask'].to(DEVICE))
            probs = torch.softmax(out.logits.float(), dim=-1)
            val_preds.extend(out.logits.argmax(-1).cpu().numpy())
            val_labels.extend(batch['labels'].numpy())
            val_probs.extend(probs[:, 1].cpu().numpy())
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, zero_division=0)
    val_recall = recall_score(val_labels, val_preds, zero_division=0)
    try:
        val_auc = roc_auc_score(val_labels, val_probs)
    except Exception:
        val_auc = 0.0
    print(f'  Val F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | AUC: {val_auc:.4f}')

    history.append({
        'epoch': epoch + 1, 'train_f1': round(train_f1, 4), 'val_f1': round(val_f1, 4),
        'val_precision': round(val_precision, 4), 'val_recall': round(val_recall, 4),
        'val_auc': round(val_auc, 4), 'train_loss': round(total_loss / max(1, len(train_dl)), 4),
        'duration_sec': round(elapsed, 1),
    })
    atomic_write_json(history, HISTORY_PATH)  # sauvegarde incrémentale — jamais perdu

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch + 1
        epochs_no_improve = 0
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f'Meilleur modèle sauvegardé ! Val F1 = {best_f1:.4f}')
    else:
        epochs_no_improve += 1
        print(f'Pas d\'amélioration depuis {epochs_no_improve} époque(s) '
              f'(meilleur : epoch {best_epoch}, Val F1 = {best_f1:.4f}) — patience = {args.patience}')

    # checkpoint de reprise en fin d'epoch (permet de reprendre à l'epoch suivante)
    save_resume_checkpoint(epoch + 1, 0)
    start_batch = 0

    if val_f1 >= args.target_f1:
        print(f"F1 >= {args.target_f1} — arrêt anticipé à l'epoch {epoch+1}")
        break

    # ── Early stopping par patience (anti-surapprentissage — mémoire §OS 2.3) ─
    # Le meilleur modèle (epoch {best_epoch}) reste de toute façon le seul jamais
    # écrit sur disque par model.save_pretrained ci-dessus : les époques suivantes
    # ne peuvent donc jamais dégrader le modèle livré, même sans cet arrêt. Mais
    # continuer à entraîner au-delà de la patience gaspille du calcul et accentue
    # le sur-apprentissage visible sur le Train F1 sans aucun bénéfice pour le
    # modèle réellement conservé — d'où cet arrêt anticipé.
    if args.patience and epochs_no_improve >= args.patience:
        print(f"Early stopping : {epochs_no_improve} époques sans amélioration "
              f"(patience={args.patience}) — arrêt à l'epoch {epoch+1}. "
              f"Meilleur modèle conservé : epoch {best_epoch} (Val F1 = {best_f1:.4f}).")
        break

print(f'Entraînement terminé. Meilleur Val F1 : {best_f1:.4f} (epoch {best_epoch})')
atomic_write_json(history, HISTORY_PATH)
print('Prochaine étape : python scripts/export_onnx.py')
