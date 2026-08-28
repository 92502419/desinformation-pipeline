# producer/src/gdelt_client.py
import httpx, asyncio, json, hashlib
from datetime import datetime, timedelta


GDELT_API_BASE = 'https://api.gdeltproject.org/api/v2/doc/doc'


async def fetch_gdelt_articles(theme: str = 'FAKE_NEWS,DISINFORMATION',
                                max_records: int = 50) -> list:
    """Récupère les derniers articles GDELT sur la désinformation"""
    params = {
        'query': f'theme:{theme}',
        'mode': 'artlist',
        'maxrecords': max_records,
        'format': 'json',
        'timespan': '15min',  # Dernières 15 minutes
        'sort': 'DateDesc',
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(GDELT_API_BASE, params=params)
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get('articles', [])
            result = []
            for a in articles:
                try:
                    tone_val = float(a.get('tone') or 0)
                except (TypeError, ValueError):
                    tone_val = 0.0
                # Déduplication via MD5 sur url+title
                art_id = hashlib.md5(
                    (a.get('url', '') + a.get('title', '')).encode()
                ).hexdigest()
                result.append({
                    'id':              art_id,
                    'title':           a.get('title', ''),
                    'body':            a.get('seendate', ''),  # GDELT n'a pas de body complet
                    'url':             a.get('url', ''),
                    'source':          a.get('domain', 'gdelt'),
                    'source_category': 'gdelt',
                    'language':        a.get('language', 'English').lower()[:2],
                    'timestamp':       datetime.utcnow().isoformat(),
                    'gdelt_tone':      tone_val,
                })
            return result
    return []

