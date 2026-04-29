# api/src/routers/web_search.py — Recherche web + classification ML en temps réel
# Flux : DuckDuckGo News → ONNX (inférence directe) → MongoDB + Kafka (apprentissage)
import os, uuid, json, logging, time
from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timezone
from pymongo import MongoClient

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["web-search"])

KAFKA_BROKER  = os.getenv("KAFKA_BROKER", "kafka:29092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC_RAW", "raw-news-stream")
MONGO_URI     = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
MONGO_DB      = os.getenv("MONGO_DB", "disinformation_db")
ONNX_DIR      = os.getenv("ONNX_MODEL_DIR", "/app/models/onnx")

# ── Cache in-memory (5 min TTL) — évite les rate-limits DuckDuckGo ──────────
_CACHE_TTL    = 300  # secondes
_search_cache: dict = {}   # { query_key: (timestamp, results) }

def _cache_get(key: str):
    if key in _search_cache:
        ts, data = _search_cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
        del _search_cache[key]
    return None

def _cache_set(key: str, data):
    # Garder max 50 entrées en mémoire
    if len(_search_cache) >= 50:
        oldest = min(_search_cache, key=lambda k: _search_cache[k][0])
        del _search_cache[oldest]
    _search_cache[key] = (time.time(), data)

# ── Chargement lazy du modèle ONNX (une seule fois au premier appel) ─────────
_ort_session  = None
_tokenizer    = None

def _get_model():
    global _ort_session, _tokenizer
    if _ort_session is None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        model_path = os.path.join(ONNX_DIR, "model_quantized.onnx")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Modèle ONNX introuvable : {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        _ort_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR, local_files_only=True)
        log.info("[web-search] Modèle ONNX chargé depuis %s", ONNX_DIR)
    return _ort_session, _tokenizer


def _classify(texts: list[str]) -> list[dict]:
    """Classe une liste de textes via ONNX. Retourne [{is_fake, confidence, p_fake}]."""
    import numpy as np
    session, tokenizer = _get_model()
    results = []
    for text in texts:
        enc = tokenizer(
            text[:512],
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in [i.name for i in session.get_inputs()]:
            inputs["token_type_ids"] = enc.get(
                "token_type_ids", np.zeros_like(enc["input_ids"])
            ).astype(np.int64)

        logits = session.run(None, inputs)[0][0]
        probs  = _softmax(logits)
        p_fake = float(probs[1])
        results.append({
            "is_fake":    1 if p_fake >= 0.5 else 0,
            "confidence": round(float(max(probs)), 4),
            "p_fake":     round(p_fake, 4),
        })
    return results


def _softmax(x):
    import numpy as np
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _publish_to_kafka(articles: list):
    """Publie les articles vers Kafka pour apprentissage continu par Spark."""
    try:
        from confluent_kafka import Producer
        p = Producer({
            "bootstrap.servers": KAFKA_BROKER,
            "client.id":         "api-websearch",
            "message.max.bytes": 2000000,
        })
        for art in articles:
            p.produce(
                topic=KAFKA_TOPIC,
                key=art["id"],
                value=json.dumps(art, ensure_ascii=False).encode("utf-8"),
            )
        p.flush(timeout=10)
    except Exception as e:
        log.warning("[web-search] Kafka indisponible (apprentissage différé) : %s", e)


def _save_to_mongo(articles: list):
    """Persiste les articles classifiés dans MongoDB."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    try:
        coll = client[MONGO_DB].articles
        from pymongo import UpdateOne
        ops = [
            UpdateOne(
                {"id": a["id"]},
                {"$set": a},
                upsert=True,
            )
            for a in articles
        ]
        if ops:
            coll.bulk_write(ops, ordered=False)
    except Exception as e:
        log.warning("[web-search] MongoDB write error : %s", e)
    finally:
        client.close()


def _google_news_search(q: str, limit: int) -> list:
    """Recherche via Google News RSS — gratuit, sans clé API, sans rate-limit."""
    import httpx, xml.etree.ElementTree as ET, urllib.parse
    url = (
        f"https://news.google.com/rss/search"
        f"?q={urllib.parse.quote(q)}&hl=fr&gl=FR&ceid=FR:fr"
    )
    try:
        resp = httpx.get(url, timeout=12, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"})
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"Erreur Google News RSS : {e}")

    root = ET.fromstring(resp.text)
    results = []
    for item in root.findall(".//item")[:limit]:
        raw_title  = item.findtext("title", "") or ""
        raw_desc   = item.findtext("description", "") or ""
        link       = item.findtext("link", "") or ""
        source_el  = item.find("source")
        source     = source_el.text if source_el is not None else "Google News"
        pubdate    = item.findtext("pubDate", "") or ""

        # Nettoyer les artefacts CDATA ou HTML éventuels
        title = raw_title.replace("<![CDATA[", "").replace("]]>", "").strip()
        body  = raw_desc.replace("<![CDATA[", "").replace("]]>", "").strip()
        # Supprimer balises HTML basiques dans la description
        import re
        body = re.sub(r"<[^>]+>", " ", body).strip()

        if title:
            results.append({
                "title":  title[:500],
                "body":   body[:2000],
                "url":    link,
                "source": source,
                "date":   pubdate,
            })
    return results


_DDG_MIN_INTERVAL = 4.0
_ddg_last_call_ts = 0.0

def _ddg_news_search(q: str, limit: int) -> list:
    """Fallback DuckDuckGo News avec délai minimum + 2 tentatives."""
    global _ddg_last_call_ts
    from duckduckgo_search import DDGS

    wait = _DDG_MIN_INTERVAL - (time.time() - _ddg_last_call_ts)
    if wait > 0:
        time.sleep(wait)

    for attempt, delay in enumerate([0, 8]):
        if delay:
            time.sleep(delay)
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(keywords=q, max_results=limit,
                                         region="wt-wt", safesearch="off"))
            _ddg_last_call_ts = time.time()
            return results
        except Exception as e:
            err_str = str(e).lower()
            if not any(x in err_str for x in ["ratelimit", "429", "403"]):
                return []  # autre erreur : ignorer silencieusement
            log.warning("[web-search] DuckDuckGo rate-limit (tentative %d/2)", attempt + 1)
    return []


def _news_search(q: str, limit: int) -> list:
    """Google News RSS (primaire) + DuckDuckGo fallback."""
    results = _google_news_search(q, limit)
    if not results:
        log.info("[web-search] Google News vide — tentative DuckDuckGo")
        results = _ddg_news_search(q, limit)
    return results


@router.get("/web")
def web_search(
    q: str    = Query(..., description="Terme à rechercher sur internet"),
    limit: int = Query(8, ge=1, le=15, description="Nombre max d'articles à analyser"),
):
    """
    Recherche des articles d'actualité sur internet via Google News RSS (primaire)
    ou DuckDuckGo News (fallback), les classifie immédiatement via ONNX DistilBERT INT8
    (< 10 ms/article) et retourne les prédictions fake/réel avec score de confiance.

    Les articles sont aussi envoyés à Kafka pour l'apprentissage continu de Spark.
    """
    # ── 1. Cache ou recherche web (Google News RSS primaire + DDG fallback) ───
    cache_key = f"{q.strip().lower()}:{limit}"
    cached = _cache_get(cache_key)
    if cached:
        log.info("[web-search] Cache hit pour «%s»", q)
        return {**cached, "cached": True}

    raw_results = _news_search(q, limit)

    if not raw_results:
        return {
            "query": q, "total_found": 0, "classified": 0,
            "articles": [],
            "message": "Aucun article trouvé pour cette requête.",
        }

    # ── 2. Mise en forme ─────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    pipeline_articles = []
    for r in raw_results:
        art_id = f"ws-{uuid.uuid4().hex[:16]}"
        pipeline_articles.append({
            "id":         art_id,
            "title":      (r.get("title")  or "")[:500],
            "body":       (r.get("body")   or "")[:2000],
            "url":        (r.get("url")    or ""),
            "source":     (r.get("source") or "web-search"),
            "language":   "fr",
            "timestamp":  (r.get("date")   or now),
            "scraped_at": now,
        })

    # ── 3. Classification ONNX directe ───────────────────────────────────────
    classified_articles = []
    try:
        texts = [
            f"{a['title']} {a['body']}".strip() for a in pipeline_articles
        ]
        preds = _classify(texts)
        for art, pred in zip(pipeline_articles, preds):
            classified_articles.append({
                **art,
                **pred,
                "processed_at": now,
                "source_type":  "web-search",
                "status":       "classified",
            })
        log.info("[web-search] %d articles classifiés via ONNX pour «%s»", len(classified_articles), q)
    except Exception as e:
        log.error("[web-search] Erreur ONNX : %s", e)
        # Fallback : retourner sans classification
        classified_articles = [
            {**a, "is_fake": None, "confidence": None, "p_fake": None, "status": "onnx_error"}
            for a in pipeline_articles
        ]

    # ── 4. Persistence MongoDB + Kafka (apprentissage continu) ───────────────
    _save_to_mongo(classified_articles)
    _publish_to_kafka(pipeline_articles)  # sans les champs ML pour éviter les conflits Spark

    n_fake = sum(1 for a in classified_articles if a.get("is_fake") == 1)
    n_real = sum(1 for a in classified_articles if a.get("is_fake") == 0)

    response = {
        "query":        q,
        "total_found":  len(raw_results),
        "classified":   len([a for a in classified_articles if a.get("status") == "classified"]),
        "fake_count":   n_fake,
        "real_count":   n_real,
        "articles":     classified_articles,
        "cached":       False,
        "message": (
            f"{len(classified_articles)}/{len(raw_results)} articles classifiés en temps réel. "
            f"Désinformation détectée : {n_fake} | Fiables : {n_real}. "
            f"Intégrés à l'apprentissage continu du modèle."
        ),
    }
    _cache_set(cache_key, response)
    return response


@router.get("/web/sources")
def get_indexed_sources():
    """Retourne la liste des sources déjà indexées dans MongoDB."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    try:
        sources = client[MONGO_DB].articles.distinct("source")
        return {"sources": sorted(sources), "count": len(sources)}
    finally:
        client.close()
