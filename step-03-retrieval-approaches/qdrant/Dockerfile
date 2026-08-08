FROM qdrant/qdrant:v1.15.5 AS qdrant

FROM python:3.12-slim

COPY --from=qdrant /qdrant/qdrant /usr/local/bin/qdrant
COPY requirements.txt /tmp/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev libunwind8 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY qdrant_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
