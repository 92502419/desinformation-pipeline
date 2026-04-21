# api/src/models/schemas.py — Schémas Pydantic pour l'API
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ArticleOut(BaseModel):
    id: str
    title: str
    body: Optional[str] = ''
    url: Optional[str] = ''
    source: Optional[str] = ''
    source_category: Optional[str] = ''
    language: Optional[str] = 'en'
    timestamp: Optional[str] = None
    processed_at: Optional[str] = None
    is_fake: int                     # 0 = réel, 1 = fake
    confidence: float
    p_fake: float
    gdelt_tone: Optional[float] = 0.0
    drift_score: Optional[float] = 0.0
    drift_active: Optional[bool] = False


class DriftEventOut(BaseModel):
    timestamp: str
    composite_score: float
    signals: dict
    drift_confirmed: bool
    recommended_lr: float
    messages_total: int
    confidence_mean_last100: float
    total_drift_events: int


class StatsOut(BaseModel):
    total_articles: int
    fake_articles: int
    real_articles: int
    fake_rate: float
    drift_events: int
    articles_last_hour: int
    timestamp: str


class SearchResult(BaseModel):
    results: list
    total: int


class ViralityPoint(BaseModel):
    hour: str
    total: int
    fakes: int
    avg_confidence: float
