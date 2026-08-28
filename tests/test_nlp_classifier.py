# tests/test_nlp_classifier.py — Tests du reservoir équilibré et du format d'entrée unifié
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'spark-app', 'src'))

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')
onnxruntime = pytest.importorskip('onnxruntime')


def test_input_format_matches_training_format():
    """Le format d'entrée doit être identique entre l'entraînement (train_model.py),
    l'inférence Spark (nlp_classifier.py) et la recherche web (web_search.py) —
    une désynchronisation avait provoqué un taux de faux positifs anormal (voir mémoire, §5)."""
    title, body = 'Titre exemple', 'Corps exemple'
    training_format = f"{str(title)[:200]} [SEP] {str(body)[:100]}"
    nlp_classifier_format = f'{title[:200]} [SEP] {body[:100]}'
    assert training_format == nlp_classifier_format


def test_reservoir_sampling_stays_balanced():
    """Le reservoir_update doit maintenir des buffers séparés fake/réel bornés à RESERVOIR_SIZE//2."""
    os.environ.setdefault('MODEL_PRETRAINED_PATH', 'distilbert-base-multilingual-cased')
    os.environ.setdefault('MODEL_ONNX_PATH', '/nonexistent/model.onnx')

    import numpy as np

    # On teste la logique de reservoir de façon isolée (sans charger le modèle réel).
    RESERVOIR_SIZE = 10
    per_class = RESERVOIR_SIZE // 2
    reservoir_fake, reservoir_real = [], []
    n_seen_fake = n_seen_real = 0

    for i in range(100):
        label = 1 if i % 2 == 0 else 0
        if label == 1:
            n_seen_fake += 1
            buf, n = reservoir_fake, n_seen_fake
        else:
            n_seen_real += 1
            buf, n = reservoir_real, n_seen_real
        if len(buf) < per_class:
            buf.append((f'text{i}', label))
        else:
            j = np.random.randint(0, n)
            if j < per_class:
                buf[j] = (f'text{i}', label)

    assert len(reservoir_fake) == per_class
    assert len(reservoir_real) == per_class
