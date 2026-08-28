# tests/test_api_health.py — Test d'intégration léger de l'API FastAPI (nécessite les
# services Docker démarrés : mongodb, elasticsearch). Ignoré automatiquement sinon.
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'api', 'src'))

fastapi_testclient = pytest.importorskip('fastapi.testclient')


@pytest.fixture(scope='module')
def client():
    try:
        from main import app
    except Exception as e:
        pytest.skip(f'API non importable (dépendances/services manquants) : {e}')
    return fastapi_testclient.TestClient(app)


def test_health_endpoint_returns_200(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    body = resp.json()
    assert 'status' in body and 'mongo' in body and 'elasticsearch' in body


def test_stats_endpoint_returns_expected_keys(client):
    resp = client.get('/api/v1/stats')
    assert resp.status_code == 200
    body = resp.json()
    for key in ('total_articles', 'fake_articles', 'real_articles', 'fake_rate', 'drift_events'):
        assert key in body
