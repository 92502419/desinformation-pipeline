# producer/src/kafka_producer.py — Producteur Kafka complet
import asyncio, feedparser, json, hashlib, logging, os
from datetime import datetime
from confluent_kafka import Producer
from gdelt_client import fetch_gdelt_articles
from rss_sources import ALL_SOURCES
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:29092')  # port interne Docker : 29092
TOPIC        = os.getenv('KAFKA_TOPIC_RAW', 'raw-news-stream')
RSS_INTERVAL = int(os.getenv('RSS_SCRAPE_INTERVAL_SEC', 60))
GDELT_INTERVAL = int(os.getenv('GDELT_QUERY_INTERVAL_SEC', 900))


# ── PRODUCTEUR KAFKA ─────────────────────────────────────
producer = Producer({
    'bootstrap.servers': KAFKA_BROKER,
    'client.id': 'rss-scraper-v2',
    'message.max.bytes': 2000000,
    'compression.type': 'gzip',
    'linger.ms': 500,          # Batch messages pour réduire la charge réseau
    'batch.size': 131072,
})


def delivery_callback(err, msg):
    if err: log.error(f'Kafka delivery FAILED: {err}')


seen_ids = set()  # Cache des articles déjà envoyés


def send_article(article: dict):
    art_id = hashlib.md5(
        (article.get('url','') + article.get('title','')).encode()
    ).hexdigest()
    if art_id in seen_ids: return 0
    seen_ids.add(art_id)
    # Garder seulement les 100 000 derniers IDs en mémoire
    if len(seen_ids) > 100000: seen_ids.pop()
    article['id'] = art_id
    producer.produce(
        topic=TOPIC, key=art_id,
        value=json.dumps(article, ensure_ascii=False).encode('utf-8'),
        callback=delivery_callback
    )
    return 1


async def scrape_rss():
    total = 0
    for src in ALL_SOURCES:
        try:
            feed = feedparser.parse(src['url'])
            for entry in feed.entries[:20]:
                article = {
                    'title': entry.get('title', ''),
                    'body':  entry.get('summary', '')[:500],
                    'url':   entry.get('link', ''),
                    'source': src['name'],
                    'source_category': src.get('category','reliable'),
                    'language': src.get('lang', 'en'),
                    'timestamp': datetime.utcnow().isoformat(),
                    'gdelt_tone': 0.0,
                }
                total += send_article(article)
        except Exception as e:
            log.warning(f'Erreur scraping {src["name"]}: {e}')
    producer.flush()
    log.info(f'RSS : {total} nouveaux articles envoyés vers Kafka')


async def scrape_gdelt():
    articles = await fetch_gdelt_articles()
    count = sum(send_article(a) for a in articles)
    producer.flush()
    log.info(f'GDELT : {count} articles envoyés')


async def main():
    log.info(f'Producteur Kafka démarré. Broker: {KAFKA_BROKER} | Topic: {TOPIC}')
    rss_counter = 0
    while True:
        await scrape_rss()
        rss_counter += RSS_INTERVAL
        if rss_counter >= GDELT_INTERVAL:
            await scrape_gdelt()
            rss_counter = 0
        await asyncio.sleep(RSS_INTERVAL)


if __name__ == '__main__':
    asyncio.run(main())

