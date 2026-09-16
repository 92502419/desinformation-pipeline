# tests/test_shared_modules.py — Garantit que les fichiers partagés Spark/API restent synchronisés
#
# calibration.py existe en deux copies (spark-app/src/calibration.py et
# api/src/calibration.py) car les deux services sont construits par des Dockerfiles
# indépendants qui ne partagent pas de volume de code (cf. docker-compose.yml).
# calibration.py::ProbabilityCalibrator l'annonce dans son docstring — ce test
# fait tenir la promesse : toute divergence entre les deux copies fait échouer la
# suite avant que la production ne tourne avec deux comportements de calibration
# différents entre le classifieur Spark et la recherche web de l'API.
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel_path):
    with open(os.path.join(ROOT, rel_path), encoding='utf-8') as fh:
        return fh.read()


def test_calibration_module_identical_in_spark_and_api():
    spark_copy = _read('spark-app/src/calibration.py')
    api_copy = _read('api/src/calibration.py')
    assert spark_copy == api_copy, (
        'spark-app/src/calibration.py et api/src/calibration.py ont divergé — '
        'reporter le correctif dans les deux fichiers (aucun des deux ne doit '
        "importer l'autre : les Dockerfiles spark-app/ et api/ ne partagent pas "
        'de volume de code, cf. docker-compose.yml).'
    )


def test_input_text_format_identical_across_inference_paths():
    """Le format '{title[:200]} [SEP] {body[:100]}' doit être strictement identique
    entre l'entraînement, l'inférence Spark et la recherche web de l'API — une
    désynchronisation avait provoqué un taux de faux positifs anormal (voir
    reports/CORRECTIONS_MEMOIRE.md et tests/test_nlp_classifier.py)."""
    sources = {
        'spark-app/src/nlp_classifier.py': "f'{title[:200]} [SEP] {body[:100]}'",
        'api/src/routers/web_search.py': 'f"{a[\'title\'][:200]} [SEP] {a[\'body\'][:100]}".strip()',
        'scripts/calibrate_model.py': "f'{str(t)[:200]} [SEP] {str(b)[:100]}'",
    }
    for rel_path, expected_snippet in sources.items():
        content = _read(rel_path)
        assert '[:200]' in content and '[SEP]' in content and '[:100]' in content, (
            f'{rel_path} ne reproduit plus le format de texte attendu ({expected_snippet!r}).'
        )
