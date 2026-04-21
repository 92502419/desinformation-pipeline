#!/usr/bin/env python3
# scripts/train_model.py — Pré-entraînement Continual-DistilBERT
# CORRECTIONS v2 : .float() sur les poids, .long() sur les labels, batch=16 CPU
# Usage (RACINE du projet, venv_main activé) :
#   python scripts/train_model.py --epochs 10 --batch_size 16 --lr 2e-5


import argparse, os, sys, json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


# ── Vérification répertoire racine ──────────────────────────
if not os.path.exists('scripts/train_model.py'):
    print('ERREUR : Lancer depuis la RACINE : cd ~/desinformation-pipeline')
    sys.exit(1)


# ── Arguments ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',     type=int,   default=10)
parser.add_argument('--batch_size', type=int,   default=16,  help='16 recommandé sur CPU')
parser.add_argument('--lr',         type=float, default=2e-5)
parser.add_argument('--max_len',    type=int,   default=128)
parser.add_argument('--model_name', default='distilbert-base-multilingual-cased')
parser.add_argument('--output_dir', default='models/pretrained')
parser.add_argument('--train_csv',  default='data/processed/train/train.csv')
parser.add_argument('--val_csv',    default='data/processed/val/val.csv')
args = parser.parse_args()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Utilisation de : {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU : {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  CPU détecté → batch_size=16 recommandé. Google Colab pour accélérer.')


# ── Vérification fichiers CSV ────────────────────────────────
for path in [args.train_csv, args.val_csv]:
    if not os.path.exists(path):
        print(f'ERREUR : {path} introuvable.')
        print('Exécuter d\'abord : python scripts/preprocess_data.py')
        sys.exit(1)


# ── Dataset ─────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.df = pd.read_csv(csv_path).fillna('')
        self.df['label'] = self.df['label'].astype(int)  # Forcer int
        self.tokenizer = tokenizer
        self.max_len   = max_len


    def __len__(self): return len(self.df)


    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = f"{str(row['title'])[:200]} [SEP] {str(row['body'])[:100]}"
        enc  = self.tokenizer(text, max_length=self.max_len,
                              padding='max_length', truncation=True,
                              return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            # ✅ FIX : dtype=torch.long OBLIGATOIRE pour CrossEntropyLoss
            'labels': torch.tensor(int(row['label']), dtype=torch.long)
        }


# ── Chargement tokenizer & modèle ────────────────────────────
print(f'Chargement du tokenizer : {args.model_name}')
tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
model     = DistilBertForSequenceClassification.from_pretrained(
    args.model_name, num_labels=2).to(DEVICE)


# ── DataLoaders ─────────────────────────────────────────────
print('Chargement des données...')
train_ds = NewsDataset(args.train_csv, tokenizer, args.max_len)
val_ds   = NewsDataset(args.val_csv,   tokenizer, args.max_len)
# Sur CPU : num_workers=0 évite les erreurs de multiprocessing
n_workers = 0 if DEVICE == 'cpu' else 4
train_dl  = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=n_workers, pin_memory=(DEVICE=='cuda'))
val_dl    = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False,
                       num_workers=n_workers)
print(f'Train : {len(train_ds)} ex. | Val : {len(val_ds)} ex.')


# ── Poids de classes ─────────────────────────────────────────
lbl_series = pd.read_csv(args.train_csv)['label'].astype(int)
n_fake  = (lbl_series == 1).sum()
n_real  = (lbl_series == 0).sum()
n_total = len(lbl_series)
print(f'fake={n_fake} ({n_fake/n_total*100:.1f}%) | real={n_real} ({n_real/n_total*100:.1f}%)')
# ✅ FIX : .float() OBLIGATOIRE — CrossEntropyLoss attend Float32
# Sans .float() → RuntimeError: expected scalar type Float but found Double
w       = torch.tensor([n_fake/n_total, n_real/n_total]).float().to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=w)


# ── Optimiseur ──────────────────────────────────────────────
layer_lrs = []
for i, layer in enumerate(model.distilbert.transformer.layer):
    layer_lrs.append({'params': layer.parameters(), 'lr': args.lr * (0.9 ** (5-i))})
layer_lrs.append({'params': model.pre_classifier.parameters(), 'lr': args.lr})
layer_lrs.append({'params': model.classifier.parameters(),     'lr': args.lr})
optimizer = AdamW(layer_lrs, weight_decay=0.01)


# ── Boucle d'entraînement ────────────────────────────────────
os.makedirs(args.output_dir, exist_ok=True)
best_f1, history = 0.0, []


for epoch in range(args.epochs):
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=max(1, len(train_dl)//5))
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in tqdm(train_dl, desc=f'Epoch {epoch+1}/{args.epochs}'):
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        # ✅ FIX : .long() OBLIGATOIRE — CrossEntropyLoss attend int64
        lbls      = batch['labels'].to(DEVICE).long()
        optimizer.zero_grad()
        out  = model(input_ids=input_ids, attention_mask=attn_mask)
        loss = loss_fn(out.logits, lbls)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds_all.extend(out.logits.argmax(-1).cpu().numpy())
        labels_all.extend(lbls.cpu().numpy())
    train_f1 = f1_score(labels_all, preds_all, average='macro')
    print(f'Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f} | Train F1: {train_f1:.4f}')


    # ── Validation ──────────────────────────────────────────
    model.eval()
    val_preds, val_labels, val_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_dl, desc='Validation', leave=False):
            out   = model(input_ids=batch['input_ids'].to(DEVICE),
                          attention_mask=batch['attention_mask'].to(DEVICE))
            probs = torch.softmax(out.logits, dim=-1)
            val_preds.extend(out.logits.argmax(-1).cpu().numpy())
            val_labels.extend(batch['labels'].numpy())
            val_probs.extend(probs[:, 1].cpu().numpy())
    val_f1  = f1_score(val_labels, val_preds, average='macro')
    try:    val_auc = roc_auc_score(val_labels, val_probs)
    except: val_auc = 0.0
    print(f'         Val F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
    history.append({'epoch': epoch+1, 'train_f1': round(train_f1,4), 'val_f1': round(val_f1,4)})


    if val_f1 > best_f1:
        best_f1 = val_f1
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f'         ✅ Meilleur modèle sauvegardé ! Val F1 = {best_f1:.4f}')
    if val_f1 >= 0.93:
        print(f'🎯 F1 >= 0.93 — arrêt anticipé à l\'epoch {epoch+1}')
        break


print(f'Entraînement terminé. Meilleur Val F1 : {best_f1:.4f}')
json.dump(history, open(f'{args.output_dir}/training_history.json','w'), indent=2)
print('Prochaine étape : python scripts/export_onnx.py')

