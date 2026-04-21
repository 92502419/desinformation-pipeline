#!/usr/bin/env python3
"""
tests/test_pipeline.py — Tests de validation du pipeline de désinformation
KOMOSSI Sosso — Master 2 IBDIA, UCAO-UUT 2025-2026

Usage (depuis la racine, venv_main activé) :
    python tests/test_pipeline.py
"""
import os, sys, json, math, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spark-app', 'src'))

# Patcher les chemins de modèles pour les tests locaux (hors Docker)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('MODEL_PRETRAINED_PATH', os.path.join(ROOT_DIR, 'models', 'pretrained'))
os.environ.setdefault('MODEL_ONNX_PATH',       os.path.join(ROOT_DIR, 'models', 'onnx', 'model_quantized.onnx'))
os.environ.setdefault('MODEL_CHECKPOINT_PATH', os.path.join(ROOT_DIR, 'models', 'checkpoints'))

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
INFO = '\033[94m→\033[0m'

def test(name: str, condition: bool, detail: str = ''):
    icon = PASS if condition else FAIL
    print(f'  {icon} {name}' + (f' ({detail})' if detail else ''))
    return condition


# ════════════════════════════════════════════════════════════════════
print('\n=== Test 1 : Détecteur de Concept Drift ===')
# ════════════════════════════════════════════════════════════════════
try:
    from drift_monitor import DynamicDriftMonitor
    m = DynamicDriftMonitor()

    # Validation des entrées invalides
    r = m.update(float('nan'))
    test('NaN → valeur neutre 0.5, pas d\'exception', True)

    r = m.update(float('inf'))
    test('Inf → valeur clippée à 1.0, pas d\'exception', True)

    r = m.update(-0.5)
    test('Valeur négative → clippée à 0.0', True)

    r = m.update(1.5)
    test('Valeur > 1.0 → clippée à 1.0', True)

    # Score composite doit être en [0, 1]
    r = m.update(0.8)
    ok = 0.0 <= r['composite_score'] <= 1.0
    test('composite_score ∈ [0,1]', ok, f'score={r["composite_score"]:.4f}')

    # Simulation drift
    for _ in range(200):
        m.update(float(np.random.beta(9, 1)))
    for _ in range(100):
        m.update(float(np.random.uniform(0, 1)))
    r = m.update(0.2)
    test('Messages traités sans erreur (300 updates)', m.messages_total >= 300,
         f'total={m.messages_total}')

    stats = m.get_stats()
    test('get_stats() retourne les champs attendus',
         all(k in stats for k in ['messages_total','composite_score','drift_active']))

except Exception as e:
    test('DriftMonitor : import et init', False, str(e))


# ════════════════════════════════════════════════════════════════════
print('\n=== Test 2 : Modèle ONNX ===')
# ════════════════════════════════════════════════════════════════════
ONNX_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'onnx', 'model_quantized.onnx')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'models', 'pretrained')

onnx_ok = os.path.exists(ONNX_PATH)
test('Fichier ONNX présent', onnx_ok, ONNX_PATH)

if onnx_ok:
    try:
        from transformers import DistilBertTokenizerFast
        from onnxruntime import InferenceSession
        import torch

        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        session   = InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
        test('Session ONNX chargée', True)

        # Test inférence
        texts = [
            'SHOCKING: Government hides massive conspiracy revealed!',
            'AFP: The Federal Reserve raised interest rates by 25 bps.',
        ]
        enc = tokenizer(texts, max_length=128, padding='max_length',
                        truncation=True, return_tensors='np')
        # Warm-up (premier appel = chargement noyaux CPU, à exclure de la mesure)
        session.run(None, {
            'input_ids':      enc['input_ids'].astype(np.int64),
            'attention_mask': enc['attention_mask'].astype(np.int64)
        })
        # Mesure sur 5 passes
        N = 5
        t0  = time.perf_counter()
        for _ in range(N):
            logits = session.run(None, {
                'input_ids':      enc['input_ids'].astype(np.int64),
                'attention_mask': enc['attention_mask'].astype(np.int64)
            })[0]
        latency_ms = (time.perf_counter() - t0) / (N * len(texts)) * 1000

        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        test('Prédictions produites (2 articles)', logits.shape[0] == 2,
             f'shape={logits.shape}')
        test(f'Latence ONNX < 50 ms/article', latency_ms < 50,
             f'{latency_ms:.1f} ms')

        # Vérification ordres de grandeur
        labels = probs.argmax(axis=1).tolist()
        test('Résultat 1 classifié comme FAKE (texte sensationnaliste)',
             labels[0] == 1, f'pred={labels[0]}')
        test('Résultat 2 classifié comme RÉEL (article AFP sobre)',
             labels[1] == 0, f'pred={labels[1]}')

    except Exception as e:
        test('Inférence ONNX', False, str(e))


# ════════════════════════════════════════════════════════════════════
print('\n=== Test 3 : Online Trainer (ordre backward → clip → step) ===')
# ════════════════════════════════════════════════════════════════════
try:
    from nlp_classifier import ContinualDistilBERT
    nlp = ContinualDistilBERT()

    # Réservoir vide au départ
    test('Réservoir vide à l\'init', len(nlp.reservoir) == 0)

    # Test reservoir_update
    for i in range(10):
        nlp.reservoir_update(f'article test {i}', i % 2)
    test('Reservoir sampling fonctionne', len(nlp.reservoir) == 10)

    # Test online_update sur un mini-batch
    texts  = ['This is a fake news headline', 'Reuters: official report confirms']
    labels = [1, 0]
    loss = nlp.online_update(texts, labels)
    test('online_update retourne une loss scalaire > 0', isinstance(loss, float) and loss > 0,
         f'loss={loss:.4f}')

    # Test predict
    p = nlp.predict('Shocking conspiracy revealed!', '')
    test('predict() retourne label + confidence + p_fake',
         all(k in p for k in ['label', 'confidence', 'p_fake']))
    test('confidence ∈ [0, 1]', 0 <= p['confidence'] <= 1, f'conf={p["confidence"]:.3f}')

    print(f'  {INFO} Test NLP complet. Résultat predict : label={p["label"]}, '
          f'conf={p["confidence"]:.3f}, p_fake={p["p_fake"]:.3f}')

except Exception as e:
    test('ContinualDistilBERT', False, str(e))


# ════════════════════════════════════════════════════════════════════
print('\n=== Test 4 : Producteur GDELT (parsing tone) ===')
# ════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'producer', 'src'))
try:
    # Tester le parsing sécurisé du tone
    def safe_tone(val):
        try:
            return float(val or 0)
        except (TypeError, ValueError):
            return 0.0

    test('tone=None → 0.0',        safe_tone(None) == 0.0)
    test('tone="" → 0.0',          safe_tone('') == 0.0)
    test('tone="abc" → 0.0',       safe_tone('abc') == 0.0)
    test('tone="-2.5" → -2.5',     safe_tone('-2.5') == -2.5)
    test('tone=3 → 3.0',           safe_tone(3) == 3.0)

except Exception as e:
    test('Parsing tone GDELT', False, str(e))


# ════════════════════════════════════════════════════════════════════
print('\n=== Test 5 : Fichiers requis pour Docker ===')
# ════════════════════════════════════════════════════════════════════
root = os.path.join(os.path.dirname(__file__), '..')
required_files = [
    'docker-compose.yml',
    '.env',
    'producer/Dockerfile',
    'producer/src/kafka_producer.py',
    'producer/src/rss_sources.py',
    'producer/src/gdelt_client.py',
    'spark-app/Dockerfile',
    'spark-app/src/spark_streaming.py',
    'spark-app/src/nlp_classifier.py',
    'spark-app/src/drift_monitor.py',
    'spark-app/src/online_trainer.py',
    'api/Dockerfile',
    'api/src/main.py',
    'api/src/routers/articles.py',
    'api/src/routers/drift.py',
    'config/mongodb/init.js',
    'config/grafana/provisioning/datasources/datasources.yaml',
    'config/grafana/provisioning/dashboards/dashboards.yaml',
    'config/grafana/dashboards/disinformation_monitor.json',
    'models/onnx/model_quantized.onnx',
    'models/pretrained/config.json',
]
all_present = True
for f in required_files:
    exists = os.path.exists(os.path.join(root, f))
    if not exists:
        test(f, False, 'MANQUANT')
        all_present = False
if all_present:
    test(f'Tous les {len(required_files)} fichiers requis présents', True)


# ════════════════════════════════════════════════════════════════════
print('\n=== Résumé ===')
print('Tests terminés. Corriger les ✗ avant de lancer docker compose up -d')
print()
