# api/src/routers/web_search.py — Recherche web + classification ML en temps réel
# Flux : DuckDuckGo News → ONNX (inférence directe) → MongoDB + Kafka (apprentissage)
import os, uuid, json, logging
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


@router.get("/web")
def web_search(
    q: str    = Query(..., description="Terme à rechercher sur internet"),
    limit: int = Query(8, ge=1, le=15, description="Nombre max d'articles à analyser"),
):
    """
    Recherche des articles d'actualité sur internet via DuckDuckGo News,
    les classifie immédiatement via ONNX DistilBERT INT8 (< 10 ms/article)
    et retourne les prédictions fake/réel avec score de confiance.

    Les articles sont aussi envoyés à Kafka pour l'apprentissage continu de Spark.
    """
    # ── 1. Recherche DuckDuckGo News ─────────────────────────────────────────
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw_results = list(ddgs.news(
                keywords=q,
                max_results=limit,
                region="wt-wt",
                safesearch="off",
            ))
    except ImportError:
        raise HTTPException(503, "Module duckduckgo-search non installé dans le conteneur API")
    except Exception as e:
        raise HTTPException(502, f"Erreur lors de la recherche internet : {str(e)}")

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

    return {
        "query":        q,
        "total_found":  len(raw_results),
        "classified":   len([a for a in classified_articles if a.get("status") == "classified"]),
        "fake_count":   n_fake,
        "real_count":   n_real,
        "articles":     classified_articles,
        "message": (
            f"{len(classified_articles)}/{len(raw_results)} articles classifiés en temps réel. "
            f"Désinformation détectée : {n_fake} | Fiables : {n_real}. "
            f"Intégrés à l'apprentissage continu du modèle."
        ),
    }


@router.get("/web/sources")
def get_indexed_sources():
    """Retourne la liste des sources déjà indexées dans MongoDB."""
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    try:
        sources = client[MONGO_DB].articles.distinct("source")
        return {"sources": sorted(sources), "count": len(sources)}
    finally:
        client.close()
