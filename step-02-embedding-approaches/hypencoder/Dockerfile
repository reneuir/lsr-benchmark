FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y git
ENV HF_HUB_OFFLINE=1

RUN cd /opt \
	&& git clone https://github.com/jfkback/hypencoder-paper.git hypencoder \
	&& pip install -e hypencoder

ADD requirements.txt /opt/hypencoder/

RUN pip3 install -r /opt/hypencoder/requirements.txt

