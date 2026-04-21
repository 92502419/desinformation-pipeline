# scripts/export_onnx.py — Conversion PyTorch → ONNX INT8
# Usage : python scripts/export_onnx.py


import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import os


MODEL_PATH = 'models/pretrained'
ONNX_PATH  = 'models/onnx'
os.makedirs(ONNX_PATH, exist_ok=True)


print('Chargement du modèle PyTorch...')
model = ORTModelForSequenceClassification.from_pretrained(MODEL_PATH, export=True)


print('Quantification INT8...')
import platform
arch = platform.machine().lower()
if 'arm' in arch or 'aarch' in arch:
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)
else:
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
print(f'Config quantification : {"arm64" if ("arm" in arch or "aarch" in arch) else "avx2"} (arch={platform.machine()})')
from optimum.onnxruntime import ORTQuantizer
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(save_dir=ONNX_PATH, quantization_config=qconfig)


print(f'Modèle ONNX INT8 sauvegardé dans : {ONNX_PATH}')


# Test de l'inférence ONNX
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import DistilBertTokenizerFast
import time


tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
ort_model = ORTModelForSequenceClassification.from_pretrained(ONNX_PATH)


test_text = 'BREAKING: Shocking discovery that will change everything!'
inputs = tokenizer(test_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')


N_ITER = 100
start = time.time()
for _ in range(N_ITER):
    outputs = ort_model(**inputs)
latency = (time.time() - start) / N_ITER * 1000


import torch.nn.functional as F
probs = F.softmax(torch.tensor(outputs.logits), dim=-1)
label = probs.argmax().item()


print(f'Résultat : {"FAKE" if label == 1 else "RÉEL"} (confiance : {probs.max().item():.3f})')
print(f'Latence ONNX INT8 : {latency:.2f} ms/article')
# Résultat attendu : ~5-6 ms/article

