FROM ubuntu:20.04 AS cpp-runtime

RUN cp "$(find /usr/lib -name 'libstdc++.so.6.*' -print -quit)" /libstdc++.so.6

FROM vespaengine/vespa:8

USER root

COPY --from=cpp-runtime /libstdc++.so.6 /opt/compat-lib/libstdc++.so.6

ENV HOME=/tmp
ENV LD_LIBRARY_PATH=/opt/compat-lib

RUN python3 -m ensurepip

RUN dnf install -y gcc gcc-c++ python3-devel \
    && dnf clean all

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY vespa_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py \
    && chmod a+rx /var/lib/private

USER vespa

WORKDIR /tmp

ENTRYPOINT ["/index-and-retrieve.py"]
