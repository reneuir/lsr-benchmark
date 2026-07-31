FROM rustlang/rust:nightly-slim AS ioqp-builder

ARG IOQP_COMMIT=00aff0e19cb265d55ee995afa7e51586d45e2389

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git libssl-dev pkg-config protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

COPY ioqp-small-corpora.patch /tmp/ioqp-small-corpora.patch

RUN git clone https://github.com/jmmackenzie/ioqp.git /ioqp \
    && cd /ioqp \
    && git checkout "${IOQP_COMMIT}" \
    && git apply /tmp/ioqp-small-corpora.patch \
    && cargo generate-lockfile \
    && cargo update -p console --precise 0.15.11 \
    && cargo build --locked --release --bin create --bin query

FROM python:3.12-slim

COPY --from=ioqp-builder /ioqp/target/release/create /usr/local/bin/ioqp-create
COPY --from=ioqp-builder /ioqp/target/release/query /usr/local/bin/ioqp-query
COPY --from=ioqp-builder /ioqp/LICENSE /usr/share/licenses/ioqp/LICENSE
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY ioqp_retrieval.py /index-and-retrieve.py

RUN chmod +x /index-and-retrieve.py

ENTRYPOINT ["/index-and-retrieve.py"]
