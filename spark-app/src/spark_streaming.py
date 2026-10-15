# spark-app/src/spark_streaming.py — Job Spark Structured Streaming principal
import os, json, logging, time
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType
)
from pymongo import MongoClient, UpdateOne
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from confluent_kafka import Producer as KafkaProducer
from nlp_classifier import ContinualDistilBERT
from drift_monitor import DynamicDriftMonitor
from online_trainer import OnlineTrainer
from datetime import datetime, timezone
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
log = logging.getLogger(__name__)


# ── INITIALISATION SPARK ─────────────────────────────────
# local[*] : exécution dans le driver (les objets ML/MongoDB ne sont pas sérialisables)
# Les JARs Kafka sont passés via SPARK_EXTRA_JARS ou spark.jars
_extra_jars     = os.getenv('SPARK_EXTRA_JARS', '')
_driver_memory  = os.getenv('SPARK_DRIVER_MEMORY', '1200m')
_builder = SparkSession.builder \
    .appName('DisinformationPipeline') \
    .master('local[2]') \
    .config('spark.streaming.stopGracefullyOnShutdown', 'true') \
    .config('spark.sql.shuffle.partitions', '2') \
    .config('spark.driver.memory', _driver_memory) \
    .config('spark.driver.maxResultSize', '256m') \
    .config('spark.driver.extraJavaOptions',
            '-XX:+UseG1GC -XX:G1HeapRegionSize=4m '
            '-XX:MaxGCPauseMillis=500 -XX:InitiatingHeapOccupancyPercent=35 '
            '-XX:+ExplicitGCInvokesConcurrent '
            '-Djava.net.preferIPv4Stack=true') \
    .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
    .config('spark.sql.streaming.forceDeleteTempCheckpointLocation', 'true') \
    .config('spark.python.worker.memory', '256m') \
    .config('spark.memory.fraction', '0.6') \
    .config('spark.memory.storageFraction', '0.3') \
    .config('spark.ui.enabled', 'true') \
    .config('spark.ui.port', '4040') \
    .config('spark.ui.showConsoleProgress', 'false') \
    .config('spark.sql.streaming.metricsEnabled', 'true') \
    .config('spark.network.timeout', '800s') \
    .config('spark.executor.heartbeatInterval', '60s')

if _extra_jars:
    _builder = _builder.config('spark.jars', _extra_jars)

spark = _builder.getOrCreate()


spark.sparkContext.setLogLevel('WARN')


# ── SCHÉMA DES MESSAGES KAFKA ────────────────────────────
NEWS_SCHEMA = StructType([
    StructField('id', StringType()),
    StructField('title', StringType()),
    StructField('body', StringType()),
    StructField('url', StringType()),
    StructField('source', StringType()),
    StructField('source_category', StringType()),
    StructField('language', StringType()),
    StructField('timestamp', StringType()),
    StructField('gdelt_tone', FloatType()),
])


# ── LECTURE DEPUIS KAFKA ─────────────────────────────────
df_raw = spark.readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', os.getenv('KAFKA_BROKER', 'kafka:29092')) \
    .option('subscribe', os.getenv('KAFKA_TOPIC_RAW', 'raw-news-stream')) \
    .option('startingOffsets', 'earliest') \
    .option('maxOffsetsPerTrigger', '100') \
    .option('failOnDataLoss', 'false') \
    .load()


df_parsed = df_raw \
    .select(from_json(col('value').cast('string'), NEWS_SCHEMA).alias('d')) \
    .select('d.*') \
    .filter(col('title').isNotNull() & (col('title') != ''))


# ── INITIALISATION DES MODULES (dans le driver) ──────────
nlp_model     = ContinualDistilBERT()
drift_monitor = DynamicDriftMonitor()
trainer       = OnlineTrainer(nlp_model, drift_monitor)


# Connexions MongoDB et Elasticsearch
mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://mongodb:27017'))
db           = mongo_client[os.getenv('MONGO_DB', 'disinformation_db')]
es_client    = Elasticsearch(
    os.getenv('ES_HOST', 'http://elasticsearch:9200'),
    request_timeout=60,
    retry_on_timeout=True,
    max_retries=3,
)


# Producteur Kafka pour les alertes de drift
drift_producer = KafkaProducer({'bootstrap.servers': os.getenv('KAFKA_BROKER', 'kafka:29092')})


def _iso_timestamp(value, default):
    """Normalise la date de publication en ISO 8601.

    La recherche web (Google News RSS) transmet des dates RFC 2822
    (« Sat, 29 Oct 2022 07:00:00 GMT ») que le champ `timestamp` (type date)
    d'Elasticsearch rejette : l'article était écrit dans MongoDB mais pas indexé,
    d'où un écart entre Streamlit (MongoDB) et Grafana (Elasticsearch).
    """
    if not value:
        return default
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).isoformat()
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return default


# ── TRAITEMENT PAR BATCH ──────────────────────────────────
def process_batch(batch_df, batch_id):
    t0 = time.time()
    rows = batch_df.collect()
    if not rows:
        return
    t_collect = time.time()
    log.info(f'Batch {batch_id}: {len(rows)} articles | collect={t_collect-t0:.1f}s')

    batch_texts, batch_labels, batch_docs = [], [], []
    now_iso = datetime.now(timezone.utc).isoformat()

    for row in rows:
        # 1. Classification NLP (ONNX — rapide 5-6 ms)
        pred = nlp_model.predict(row.title or '', row.body or '')

        # 2. Mise à jour du moniteur de drift
        drift_result = drift_monitor.update(
            confidence=pred['p_fake'],
            error_bit=0   # Supervision partielle : on suppose correct par défaut
        )

        # 3. Construction du document enrichi
        doc = {
            'id':              row.id,
            'title':           row.title,
            'body':            row.body or '',
            'url':             row.url or '',
            'source':          row.source or '',
            'source_category': row.source_category or 'unknown',
            'language':        row.language or 'en',
            'timestamp':       _iso_timestamp(row.timestamp, now_iso),
            'processed_at':    now_iso,
            'is_fake':         pred['label'],
            'verdict':         pred['verdict'],
            'confidence':      pred['confidence'],
            'p_fake':          pred['p_fake'],
            'p_fake_raw':      pred['p_fake_raw'],
            'gdelt_tone':      float(row.gdelt_tone or 0.0),
            'drift_score':     drift_result['composite_score'],
            'drift_active':    drift_result['drift'],
        }
        batch_docs.append(doc)

        # 4. Accumuler pour l'online learning (fake ET réel)
        # On s'entraîne sur TOUS les articles (fake et réel) mais uniquement
        # lorsque la confiance est élevée (> 0.85) pour éviter de renforcer
        # les erreurs de classification sur les articles ambigus.
        # Un article classé fake avec 90% de confiance est un bon exemple d'apprentissage.
        # Un article incertain (60-80%) risque d'être mal étiqueté → skip.
        text = f"{row.title} [SEP] {(row.body or '')[:100]}"
        if pred['confidence'] > 0.85 and pred['verdict'] != 'uncertain':
            batch_texts.append(text)
            batch_labels.append(pred['label'])
            nlp_model.reservoir_update(text, pred['label'])

    t_loop = time.time()
    log.info(f'Batch {batch_id}: boucle inference={t_loop-t_collect:.1f}s ({(t_loop-t_collect)/len(rows)*1000:.0f}ms/article)')

    # 5. Online learning via OnlineTrainer
    t_train_start = time.time()
    train_metrics = trainer.step(batch_texts, batch_labels)
    t_train_end = time.time()
    log.info(
        f'Online loss: {train_metrics["loss"]:.4f} | '
        f'lr: {train_metrics["lr"]:.2e} | '
        f'drift: {drift_monitor.is_drift_active()} | '
        f'train={t_train_end-t_train_start:.1f}s'
    )

    # 5b. Récupération : le lot contient des articles réels rejoués par
    # POST /api/v1/drift/recover (ou la récupération automatique) → fin de dérive
    # explicite, tracée dans drift-events (phase=recovery) pour Grafana.
    n_recovery = sum(1 for r in rows if r.source == 'SimulationRecovery')
    if n_recovery and drift_monitor.recover():
        recovery_doc = {**drift_monitor.get_alert_payload(), 'phase': 'recovery',
                        '_batch_id': batch_id}
        log.warning(f'RÉCUPÉRATION : fin de dérive ({n_recovery} articles réels dans le lot)')
        try:
            db.drift_events.insert_one(dict(recovery_doc))
            es_client.index(index='drift-events', document=recovery_doc)
        except Exception as e:
            log.warning(f'Erreur stockage événement de récupération : {e}')

    # 6. Émission d'une alerte + stockage si dérive détectée
    if drift_monitor.is_drift_active():
        alert = {**drift_monitor.get_alert_payload(), 'phase': 'drift'}
        drift_producer.produce(
            topic=os.getenv('KAFKA_TOPIC_DRIFT', 'drift-alerts'),
            value=json.dumps(alert).encode('utf-8')
        )
        drift_producer.flush()
        log.warning(f'DRIFT ALERT : score={alert["composite_score"]:.3f} | {alert["signals"]}')
        # Persister l'événement de drift dans MongoDB
        try:
            db.drift_events.insert_one({**alert, '_batch_id': batch_id})
        except Exception as e:
            log.warning(f'Erreur stockage drift event : {e}')
        # Indexer l'événement dans Elasticsearch (lu par le panneau Grafana "Événements de Concept Drift")
        try:
            es_client.index(index='drift-events', document={**alert, '_batch_id': batch_id})
        except Exception as e:
            log.warning(f'Erreur indexation Elasticsearch drift event : {e}')

    # 7. Stockage MongoDB — upsert sur 'id' (évite les doublons)
    if batch_docs:
        ops = [
            UpdateOne({'id': d['id']}, {'$set': d}, upsert=True)
            for d in batch_docs
        ]
        try:
            result = db.articles.bulk_write(ops, ordered=False)
            log.info(f'MongoDB : {result.upserted_count} inserts | {result.modified_count} updates')
        except Exception as e:
            log.error(f'Erreur MongoDB bulk_write : {e}')

    # 8. Indexation Elasticsearch (bulk)
    if batch_docs:
        es_docs = [
            {'_index': 'articles', '_id': d['id'], '_source': d}
            for d in batch_docs
        ]
        try:
            ok, errors = bulk(es_client, es_docs, raise_on_error=False)
            if errors:
                first = next(iter(errors[0].values()), {}).get('error', {})
                log.warning(f'Elasticsearch : {len(errors)} erreurs lors du bulk insert '
                            f'(1re : {first.get("reason", first)})')
        except Exception as e:
            log.error(f'Erreur Elasticsearch bulk : {e}')

    # 9. Métriques d'apprentissage continu — un document par micro-batch, lu par
    # les panneaux Grafana « Apprentissage continu » (le flux d'entraînement était
    # invisible hors dérive : drift-events n'est écrit que pendant une dérive).
    n_fake_batch = sum(d['is_fake'] for d in batch_docs)
    metrics_doc = {
        'timestamp':        now_iso,
        'batch_id':         batch_id,
        'articles':         len(batch_docs),
        'articles_fake':    n_fake_batch,
        'articles_real':    len(batch_docs) - n_fake_batch,
        'articles_simulation_drift':    sum(1 for r in rows if r.source == 'SimulationDrift'),
        'articles_simulation_recovery': n_recovery,
        'fake_rate_batch':  n_fake_batch / len(batch_docs) if batch_docs else 0.0,
        'p_fake_mean':      sum(d['p_fake'] for d in batch_docs) / len(batch_docs) if batch_docs else 0.0,
        'confidence_mean':  sum(d['confidence'] for d in batch_docs) / len(batch_docs) if batch_docs else 0.0,
        'train_examples_batch': len(batch_texts),
        'train_examples_total': trainer.total_examples,
        'trained':          train_metrics['loss'] > 0,
        'online_loss':      train_metrics['loss'] if train_metrics['loss'] > 0 else None,
        'learning_rate':    train_metrics['lr'],
        'reservoir_fake':   len(nlp_model.reservoir_fake),
        'reservoir_real':   len(nlp_model.reservoir_real),
        'drift_score':      drift_monitor.composite_score,
        'drift_active':     drift_monitor.is_drift_active(),
        'drift_confirmed':  drift_monitor.drift_confirmed,
        'messages_total':   drift_monitor.messages_total,
        'latency_ms_per_article': (t_loop - t_collect) / len(rows) * 1000,
        # Publication du modèle appris en ligne (checkpoint + ONNX servi à l'inférence)
        'onnx_synced':      trainer.last_step_synced,
        'onnx_syncs_total': trainer.onnx_syncs,
        'batches_before_onnx_sync': -trainer.batch_count % int(os.getenv('SYNC_ONNX_EVERY', 100)),
    }
    try:
        es_client.index(index='training-metrics', document=metrics_doc)
    except Exception as e:
        log.warning(f'Erreur indexation training-metrics : {e}')


# ── LANCEMENT DU STREAMING ───────────────────────────────
_trigger_interval = os.getenv('SPARK_MICRO_BATCH_INTERVAL', '5 seconds')
query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .trigger(processingTime=_trigger_interval) \
    .option('checkpointLocation', os.getenv('SPARK_CHECKPOINT_DIR', '/tmp/spark-checkpoints')) \
    .start()


log.info('Pipeline Spark Streaming démarré. En attente de données Kafka...')
query.awaitTermination()
