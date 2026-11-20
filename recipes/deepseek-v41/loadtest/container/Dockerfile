# The cluster stress suite as one image: the tools, their pinned Python
# dependencies, kubectl for the fault phase, and the realmix dataset and its
# tokenizer. Built by build.sh, which stages the build context.
#
#   docker run --rm -e STRESS_API_KEY -v "$PWD/out:/work" IMAGE \
#       --url https://GATEWAY --phases func,needles,overload
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG KUBECTL_VERSION=v1.31.4
ARG TARGETARCH=amd64
COPY fetch_kubectl.py /tmp/fetch_kubectl.py
RUN python /tmp/fetch_kubectl.py && rm /tmp/fetch_kubectl.py

COPY requirements.txt /opt/cluster-stress/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /opt/cluster-stress/requirements.txt

COPY tools/ /opt/cluster-stress/
COPY data/ /opt/cluster-stress/data/
RUN cd /opt/cluster-stress/data && sha256sum -c SHA256SUMS && cd .. \
    && python -m unittest -q test_cluster_stress test_oracle_tools test_live_metrics \
    && rm -rf __pycache__

ARG REVISION=unknown
ENV STRESS_REVISION=${REVISION} \
    STRESS_DATASET=/opt/cluster-stress/data/oracle-realmix-256-r36-m3-modeled-long25-38k-v3.json \
    STRESS_SOURCE_TOKENIZER=/opt/cluster-stress/data/minimax-m3 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --uid 1000 --create-home stress && mkdir /work && chown stress /work
USER stress
WORKDIR /work
ENTRYPOINT ["python", "/opt/cluster-stress/cluster_stress.py"]
CMD ["--help"]
