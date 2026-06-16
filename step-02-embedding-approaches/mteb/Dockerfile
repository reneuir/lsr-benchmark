#Start from build with all dependencies
FROM webis/lsr-benchmark-submissions:sentence-transformers-step-02-em-latest-tira-docker-software-id-blazing-buoy
#FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ADD requirements.txt /tmp/

RUN apt-get update && apt-get install -y git && python3 -m pip install --no-cache -r /tmp/requirements.txt
ENV HF_HUB_OFFLINE=1

ADD mteb_embeddings.py /mteb_embeddings.py
