# spark-app/src/nlp_classifier.py — Classification ONNX + Online Learning
# v2.0 : reservoir équilibré par classe pour éviter le biais vers fake
import os, torch, numpy as np
from transformers import DistilBertTokenizerFast
from onnxruntime import InferenceSession, SessionOptions
from torch.optim import SGD
import torch.nn.functional as F


MODEL_PRETRAINED    = os.getenv('MODEL_PRETRAINED_PATH', '/app/models/pretrained')
MODEL_ONNX          = os.getenv('MODEL_ONNX_PATH', '/app/models/onnx/model_quantized.onnx')
ONLINE_LR_BASE      = float(os.getenv('ONLINE_LR_BASE', 1e-5))
RESERVOIR_SIZE      = int(os.getenv('RESERVOIR_BUFFER_SIZE', 5000))
_RESERVOIR_PER_CLASS = RESERVOIR_SIZE // 2   # 2 500 fake + 2 500 réels


class ContinualDistilBERT:
    """Continual-DistilBERT : inférence ONNX + online learning PyTorch.

    Reservoir équilibré : un buffer séparé par classe (fake / réel).
    Empêche le sur-apprentissage vers fake lors d'injections massives de drift.
    Applicable en Afrique subsaharienne (AFP, RFI, Jeune Afrique, Al Jazeera…).
    """

    def __init__(self):
        from transformers import DistilBertForSequenceClassification
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PRETRAINED)
        _opts = SessionOptions()
        _opts.intra_op_num_threads = 2
        _opts.inter_op_num_threads = 1
        self.ort_session = InferenceSession(
            MODEL_ONNX,
            sess_options=_opts,
            providers=['CPUExecutionProvider']
        )
        self.pt_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PRETRAINED)
        self.pt_model.eval()  # eval par défaut — train() uniquement pendant online_update
        self.optimizer = SGD(self.pt_model.parameters(), lr=ONLINE_LR_BASE, momentum=0.9, weight_decay=0.01)

        # Reservoirs SÉPARÉS — évite que le drift injector sature le buffer de fake
        self.reservoir_fake: list = []
        self.reservoir_real: list = []
        self.n_seen_fake = 0
        self.n_seen_real = 0
        self.current_lr  = ONLINE_LR_BASE
        print('[NLP] Continual-DistilBERT v2.0 — reservoir équilibré (fake/réel)')

    @property
    def reservoir(self):
        """Rétrocompatibilité : vue combinée des deux buffers."""
        return self.reservoir_fake + self.reservoir_real

    def predict(self, title: str, body: str = '') -> dict:
        """Inférence ONNX INT8 (~5-6 ms) — 100 % numpy, zéro tensor PyTorch."""
        text = f'{title[:200]} [SEP] {body[:100]}'
        enc  = self.tokenizer(text, max_length=128, padding='max_length',
                              truncation=True, return_tensors='np')
        logits = self.ort_session.run(None, {
            'input_ids':      enc['input_ids'].astype(np.int64),
            'attention_mask': enc['attention_mask'].astype(np.int64)
        })[0][0]  # shape (2,)
        e = np.exp(logits - logits.max())  # softmax numeriquement stable
        probs = e / e.sum()
        label = 1 if float(probs[1]) >= 0.65 else 0
        return {'label': label, 'confidence': float(probs[label]), 'p_fake': float(probs[1])}

    def reservoir_update(self, text: str, label: int):
        """Reservoir sampling équilibré : maintient 50 % fake / 50 % réel.

        Même si 80 % des articles entrants sont fake (drift injector),
        le buffer de replay reste équilibré pour résister au biais de classe.
        """
        if label == 1:
            self.n_seen_fake += 1
            buf = self.reservoir_fake
            n   = self.n_seen_fake
        else:
            self.n_seen_real += 1
            buf = self.reservoir_real
            n   = self.n_seen_real

        if len(buf) < _RESERVOIR_PER_CLASS:
            buf.append((text, label))
        else:
            j = np.random.randint(0, n)
            if j < _RESERVOIR_PER_CLASS:
                buf[j] = (text, label)

    def online_update(self, batch_texts: list, batch_labels: list, lr: float = None):
        """Mise à jour online avec rééquilibrage batch + replay 50/50."""
        if lr and lr != self.current_lr:
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            self.current_lr = lr

        # ── Rééquilibrage du batch entrant ───────────────────────────────────
        MAX_EACH = 4  # 4 fake + 4 réels max depuis le batch = 8 exemples
        fake_in  = [(t, l) for t, l in zip(batch_texts, batch_labels) if l == 1]
        real_in  = [(t, l) for t, l in zip(batch_texts, batch_labels) if l == 0]
        n_each   = min(len(fake_in), len(real_in), MAX_EACH)

        if n_each > 0:
            selected = fake_in[:n_each] + real_in[:n_each]
        else:
            all_in   = list(zip(batch_texts, batch_labels))
            selected = all_in[:MAX_EACH]

        train_texts  = [s[0] for s in selected]
        train_labels = [s[1] for s in selected]

        # ── Replay équilibré depuis les reservoirs séparés ───────────────────
        rp = min(2, len(self.reservoir_fake), len(self.reservoir_real))
        if rp > 0:
            fi = np.random.choice(len(self.reservoir_fake), rp, replace=False)
            ri = np.random.choice(len(self.reservoir_real), rp, replace=False)
            replay = ([self.reservoir_fake[i] for i in fi] +
                      [self.reservoir_real[i] for i in ri])
            train_texts  += [r[0] for r in replay]
            train_labels += [r[1] for r in replay]
        elif self.reservoir:
            idxs = np.random.choice(len(self.reservoir), min(4, len(self.reservoir)), replace=False)
            replay = [self.reservoir[i] for i in idxs]
            train_texts  += [r[0] for r in replay]
            train_labels += [r[1] for r in replay]

        # ── Loss pondérée équilibrée ──────────────────────────────────────────
        n_f  = sum(1 for l in train_labels if l == 1)
        n_r  = len(train_labels) - n_f
        if n_f > 0 and n_r > 0:
            tot = len(train_labels)
            # Poids inversés : classe minoritaire reçoit plus de signal
            cw = torch.tensor([n_f / tot, n_r / tot], dtype=torch.float)
        else:
            cw = None

        self.pt_model.train()
        enc      = self.tokenizer(train_texts, max_length=128, padding='max_length',
                                  truncation=True, return_tensors='pt')
        labels_t = torch.tensor(train_labels, dtype=torch.long)
        self.optimizer.zero_grad()
        out      = self.pt_model(**enc)
        loss     = torch.nn.CrossEntropyLoss(weight=cw)(out.logits, labels_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pt_model.parameters(), 1.0)
        self.optimizer.step()
        loss_val = float(loss.item())
        del enc, labels_t, out, loss
        self.optimizer.zero_grad(set_to_none=True)  # libère la mémoire des gradients
        self.pt_model.eval()  # repasse en eval pour libérer les buffers d'activation
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

