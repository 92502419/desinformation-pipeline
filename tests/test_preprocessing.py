# tests/test_preprocessing.py — Tests unitaires du nettoyage de texte et du schéma des CSV
import importlib.util
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_clean_text():
    spec = importlib.util.spec_from_file_location(
        'preprocess_data', os.path.join(ROOT, 'scripts', 'preprocess_data.py'))
    # Le module fait du travail au chargement (lecture de data/raw/...) — on charge
    # seulement la fonction clean_text par introspection du fichier source pour éviter
    # de dépendre de données déjà téléchargées.
    src = open(spec.origin, encoding='utf-8').read()
    func_src = src.split('def clean_text')[1].split('\ndfs = []')[0]
    ns = {}
    exec('def clean_text' + func_src, {'re': __import__('re')}, ns)
    return ns['clean_text']


clean_text = _load_clean_text()


def test_clean_text_strips_urls():
    assert '[URL]' in clean_text('Voir https://example.com/article pour plus')


def test_clean_text_strips_mentions():
    assert '[USER]' in clean_text('Merci @jdupont pour le tuyau')


def test_clean_text_handles_non_string():
    assert clean_text(None) == ''
    assert clean_text(float('nan')) == ''


def test_clean_text_truncates_to_2000_chars():
    long_text = 'a' * 5000
    assert len(clean_text(long_text)) == 2000


def test_processed_csv_schema_if_present():
    """Si le prétraitement a déjà tourné, vérifie le schéma attendu des CSV générés."""
    train_csv = os.path.join(ROOT, 'data', 'processed', 'train', 'train.csv')
    if not os.path.exists(train_csv):
        pytest.skip('data/processed/train/train.csv absent — lancer scripts/preprocess_data.py')
    df = pd.read_csv(train_csv)
    assert set(['title', 'body', 'label', 'source']).issubset(df.columns)
    assert set(df['label'].unique()).issubset({0, 1})
    assert len(df) > 0
