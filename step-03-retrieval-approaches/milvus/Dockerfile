FROM milvusdb/milvus:v2.6.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip python3-dev gcc g++ \
    && chmod -R a+rX /milvus/bin /milvus/configs /milvus/lib \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY milvus_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
