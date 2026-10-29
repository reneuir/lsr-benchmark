FROM python:3.12-slim

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY rank_fusion.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
