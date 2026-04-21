# producer/Dockerfile — VERSION CORRIGÉE
# ✅ python:3.12-slim (PAS 3.11-slim)
FROM python:3.12-slim


WORKDIR /app


# Dépendances système pour confluent-kafka (librdkafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    librdkafka-dev gcc && rm -rf /var/lib/apt/lists/*


# Upgrade pip en premier
RUN pip install --no-cache-dir --upgrade pip setuptools wheel


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ ./src/


CMD ["python", "src/kafka_producer.py"]

