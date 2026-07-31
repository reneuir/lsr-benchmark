FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --no-deps -r /tmp/requirements.txt \
    && pip3 install --no-deps ir_datasets ir_measures \
    && pip3 install --no-cache-dir faiss-cpu numpy click tirex-tracker tqdm tira \
    && rm /tmp/requirements.txt

COPY faiss_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
