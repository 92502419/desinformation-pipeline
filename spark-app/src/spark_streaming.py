# spark-app/src/spark_streaming.py — Job Spark Structured Streaming principal
import os, json, logging
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
_extra_jars = os.getenv('SPARK_EXTRA_JARS', '')
_builder = SparkSession.builder \
    .appName('DisinformationPipeline') \
    .master('local[2]') \
    .config('spark.streaming.stopGracefullyOnShutdown', 'true') \
    .config('spark.sql.shuffle.partitions', '2') \
    .config('spark.driver.memory', '1500m') \
    .config('spark.driver.maxResultSize', '256m') \
    .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
    .config('spark.sql.streaming.forceDeleteTempCheckpointLocation', 'true') \
    .config('spark.python.worker.memory', '256m') \
    .config('spark.memory.fraction', '0.6') \
    .config('spark.memory.storageFraction', '0.3') \
    .config('spark.ui.enabled', 'false') \
    .config('spark.sql.streaming.metricsEnabled', 'false') \
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
    .option('maxOffsetsPerTrigger', '1000') \
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


# ── TRAITEMENT PAR BATCH ──────────────────────────────────
def process_batch(batch_df, batch_id):
    rows = batch_df.collect()
    if not rows:
        return
    log.info(f'Batch {batch_id}: {len(rows)} articles')

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
            'timestamp':       row.timestamp or now_iso,
            'processed_at':    now_iso,
            'is_fake':         pred['label'],
            'confidence':      pred['confidence'],
            'p_fake':          pred['p_fake'],
            'gdelt_tone':      float(row.gdelt_tone or 0.0),
            'drift_score':     drift_result['composite_score'],
            'drift_active':    drift_result['drift'],
        }
        batch_docs.append(doc)

        # 4. Accumuler pour l'online learning
        text = f"{row.title} [SEP] {(row.body or '')[:100]}"
        batch_texts.append(text)
        batch_labels.append(pred['label'])
        nlp_model.reservoir_update(text, pred['label'])

    # 5. Online learning via OnlineTrainer
    train_metrics = trainer.step(batch_texts, batch_labels)
    log.info(
        f'Online loss: {train_metrics["loss"]:.4f} | '
        f'lr: {train_metrics["lr"]:.2e} | '
        f'drift: {drift_monitor.is_drift_active()}'
    )

    # 6. Émission d'une alerte + stockage si dérive détectée
    if drift_monitor.is_drift_active():
        alert = drift_monitor.get_alert_payload()
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
                log.warning(f'Elasticsearch : {len(errors)} erreurs lors du bulk insert')
        except Exception as e:
            log.error(f'Erreur Elasticsearch bulk : {e}')


# ── LANCEMENT DU STREAMING ───────────────────────────────
_trigger_interval = os.getenv('SPARK_MICRO_BATCH_INTERVAL', '5 seconds')
query = df_parsed.writeStream \
    .foreachBatch(process_batch) \
    .trigger(processingTime=_trigger_interval) \
    .option('checkpointLocation', os.getenv('SPARK_CHECKPOINT_DIR', '/tmp/spark-checkpoints')) \
    .start()


log.info('Pipeline Spark Streaming démarré. En attente de données Kafka...')
query.awaitTermination()
