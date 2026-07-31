FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY faiss_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
