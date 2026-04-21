FROM python:3.12-slim
# ─── Java 17 (requis par PySpark) + dépendances système ──────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless curl wget librdkafka-dev gcc procps && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir 'packaging>=23.2'

# ─── PyTorch CPU ──────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ─── Dépendances NLP + Big Data ───────────────────────────────────────────────
RUN pip install --no-cache-dir \
    pyspark==3.5.3 \
    transformers==4.42.4 \
    river==0.21.2 \
    onnxruntime==1.19.0 \
    "optimum[onnxruntime]==1.21.0" \
    pymongo==4.9.1 \
    "elasticsearch==8.15.0" \
    confluent-kafka==2.5.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    python-dotenv==1.0.1

# ─── JARs Kafka pour Spark 3.5.3 ─────────────────────────────────────────────
# Versions exactes requises par spark-sql-kafka 3.5.3
RUN mkdir -p /opt/spark-jars && \
    curl -fsSL https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.3/spark-sql-kafka-0-10_2.12-3.5.3.jar \
         -o /opt/spark-jars/spark-sql-kafka.jar && \
    curl -fsSL https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar \
         -o /opt/spark-jars/kafka-clients.jar && \
    curl -fsSL https://repo1.maven.org/maven2/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.3/spark-token-provider-kafka-0-10_2.12-3.5.3.jar \
         -o /opt/spark-jars/spark-token-provider-kafka.jar && \
    curl -fsSL https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar \
         -o /opt/spark-jars/commons-pool2.jar

# Indiquer à Spark où trouver les JARs supplémentaires
ENV SPARK_EXTRA_JARS=/opt/spark-jars/spark-sql-kafka.jar,/opt/spark-jars/kafka-clients.jar,/opt/spark-jars/spark-token-provider-kafka.jar,/opt/spark-jars/commons-pool2.jar

WORKDIR /app
COPY src/ ./src/

CMD ["python", "src/spark_streaming.py"]
