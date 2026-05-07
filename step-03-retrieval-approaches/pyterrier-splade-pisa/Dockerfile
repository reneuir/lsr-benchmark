FROM ubuntu:jammy


RUN apt-get update \
    && apt-get install -y \
        git \
        openjdk-11-jdk \
        python3 \
        python3-pip \
        build-essential \
        cmake \
        ninja-build \
        libtbb-dev \
        libtbb2 \
        patchelf \
        libtool \
        automake \
        autoconf \
    && rm -rf /var/lib/apt/lists/*

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5


COPY arm-patches/ /tmp/arm-patches/
RUN ARCH="$(dpkg --print-architecture)" \
    && if [ "$ARCH" = "amd64" ]; then \
        pip3 install --no-cache-dir pyterrier_pisa; \
    else \
        pip3 install --no-cache-dir scikit-build-core numpy cmake ninja \
        && git clone --recurse-submodules https://github.com/terrierteam/pyterrier_pisa.git /tmp/pyterrier_pisa \
        && cp /tmp/arm-patches/CMakeLists.txt   /tmp/pyterrier_pisa/CMakeLists.txt \
        && cp /tmp/arm-patches/requirements.txt /tmp/pyterrier_pisa/requirements.txt \
        && pip3 install --no-cache-dir /tmp/pyterrier_pisa \
        && rm -rf /tmp/pyterrier_pisa; \
    fi \
    && rm -rf /tmp/arm-patches

ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN python3 -c "import pyterrier as pt; pt.init();" \
    && apt-get update \
    && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ADD run-pyterrier-splade.py /run-pyterrier-splade.py

