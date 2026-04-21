# spark-app/src/nlp_classifier.py — Classification ONNX + Online Learning
import os, torch, numpy as np
from transformers import DistilBertTokenizerFast
from onnxruntime import InferenceSession, SessionOptions
from torch.optim import AdamW
import torch.nn.functional as F


MODEL_PRETRAINED = os.getenv('MODEL_PRETRAINED_PATH', '/app/models/pretrained')
MODEL_ONNX       = os.getenv('MODEL_ONNX_PATH', '/app/models/onnx/model_quantized.onnx')
ONLINE_LR_BASE   = float(os.getenv('ONLINE_LR_BASE', 1e-5))
RESERVOIR_SIZE   = int(os.getenv('RESERVOIR_BUFFER_SIZE', 5000))


class ContinualDistilBERT:
    """Continual-DistilBERT : inférence ONNX + online learning PyTorch"""


    def __init__(self):
        from transformers import DistilBertForSequenceClassification
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PRETRAINED)
        # Modèle ONNX pour l'inférence rapide — limité à 2 threads pour ne pas saturer le CPU
        _opts = SessionOptions()
        _opts.intra_op_num_threads = 2
        _opts.inter_op_num_threads = 1
        self.ort_session = InferenceSession(
            MODEL_ONNX,
            sess_options=_opts,
            providers=['CPUExecutionProvider']
        )
        # Modèle PyTorch pour les mises à jour online
        self.pt_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PRETRAINED)
        self.pt_model.train()
        self.optimizer = AdamW(self.pt_model.parameters(), lr=ONLINE_LR_BASE, weight_decay=0.01)
        # Reservoir buffer pour éviter l'oubli catastrophique
        self.reservoir = []
        self.n_seen = 0
        self.current_lr = ONLINE_LR_BASE
        print('[NLP] Continual-DistilBERT initialisé (ONNX + PyTorch online)')


    def predict(self, title: str, body: str = '') -> dict:
        """Inférence ONNX (5-6 ms) — NE met PAS à jour le modèle"""
        text = f'{title[:200]} [SEP] {body[:100]}'
        enc  = self.tokenizer(text, max_length=128, padding='max_length',
                              truncation=True, return_tensors='np')
        logits = self.ort_session.run(None, {
            'input_ids': enc['input_ids'].astype(np.int64),
            'attention_mask': enc['attention_mask'].astype(np.int64)
        })[0]
        probs = F.softmax(torch.tensor(logits), dim=-1).squeeze()
        label = int(probs.argmax().item())
        return {'label': label, 'confidence': float(probs[label]),'p_fake': float(probs[1])}


    def reservoir_update(self, text: str, label: int):
        """Reservoir sampling : garde un échantillon représentatif des données passées"""
        self.n_seen += 1
        if len(self.reservoir) < RESERVOIR_SIZE:
            self.reservoir.append((text, label))
        else:
            j = np.random.randint(0, self.n_seen)
            if j < RESERVOIR_SIZE:
                self.reservoir[j] = (text, label)


    def online_update(self, batch_texts: list, batch_labels: list, lr: float = None):
        """Mise à jour online sur un batch + replay du reservoir (mini-batch pour économiser la RAM)"""
        if lr and lr != self.current_lr:
            for pg in self.optimizer.param_groups: pg['lr'] = lr
            self.current_lr = lr

        # Batch réduit à 8 pour économiser RAM (DistilBERT PyTorch crée des tensors
        # [batch × heads × seq × seq] → mémoire proportionnelle au batch size)
        MAX_TRAIN_TEXTS = 8
        train_texts = batch_texts[:MAX_TRAIN_TEXTS]
        train_labels = batch_labels[:MAX_TRAIN_TEXTS]

        # Replay du reservoir (max 4 exemples pour rester à batch ≤ 12 total)
        replay_size = min(4, len(self.reservoir))
        if replay_size > 0:
            idxs = np.random.choice(len(self.reservoir), replay_size, replace=False)
            replay = [self.reservoir[i] for i in idxs]
            train_texts  = train_texts  + [r[0] for r in replay]
            train_labels = train_labels + [r[1] for r in replay]

        # Tokenisation du mini-batch
        enc = self.tokenizer(train_texts, max_length=128, padding='max_length',
                             truncation=True, return_tensors='pt')
        labels_t = torch.tensor(train_labels, dtype=torch.long)

        # Forward + backward
        self.optimizer.zero_grad()
        outputs = self.pt_model(**enc, labels=labels_t)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pt_model.parameters(), 1.0)
        self.optimizer.step()

        # Sauvegarder la loss AVANT de libérer les tensors
        loss_val = float(outputs.loss.item()) if hasattr(outputs, 'loss') else 0.0
        del enc, labels_t, outputs
        return loss_val


    def sync_onnx(self):
        """Exporte le modèle PyTorch mis à jour vers ONNX pour accélérer l'inférence"""
        import platform
        from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        # Sauvegarder le modèle PyTorch
        checkpoint_dir = os.getenv('MODEL_CHECKPOINT_PATH', '/app/models/checkpoints')
        self.pt_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        # Re-exporter en ONNX INT8 — config compatible x86_64 et arm64
        ort_model = ORTModelForSequenceClassification.from_pretrained(checkpoint_dir, export=True)
        arch = platform.machine().lower()
        if 'arm' in arch or 'aarch' in arch:
            qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
        else:
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer = ORTQuantizer.from_pretrained(ort_model)
        quantizer.quantize(save_dir=os.path.dirname(MODEL_ONNX), quantization_config=qconfig)
        # Recharger la session ONNX avec le nouveau modèle (thread limit maintenu)
        _opts = SessionOptions()
        _opts.intra_op_num_threads = 2
        _opts.inter_op_num_threads = 1
        self.ort_session = InferenceSession(MODEL_ONNX, sess_options=_opts, providers=['CPUExecutionProvider'])
        print('[NLP] Modèle ONNX resynchronisé avec les poids online')

