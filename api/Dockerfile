# api/Dockerfile — API FastAPI
# Python 3.12-slim, uvicorn multi-workers
FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Versions épinglées et compatibles Python 3.12 + Pydantic v2
RUN pip install --no-cache-dir \
    fastapi==0.113.0 \
    uvicorn[standard]==0.30.6 \
    pymongo==4.9.1 \
    elasticsearch==8.15.0 \
    python-dotenv==1.0.1 \
    pydantic==2.9.2 \
    confluent-kafka==2.5.0

COPY src/ ./src/
# scripts/ est monté en volume depuis la racine du projet (voir docker-compose.yml)

# Démarrage depuis /app (src.main:app) : sys.path inclut /app,
# et main.py ajoute /app/src via sys.path.insert pour les imports de routers
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
