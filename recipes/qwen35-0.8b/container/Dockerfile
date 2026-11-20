# syntax=docker/dockerfile:1
# Qwen3.5-0.8B on one GPU: the staging worker image for confidential-inference.
# Build with build.sh from the repository root.
#
# The Python environment is the DeepSeek-V4.1 image's (recipes/deepseek-v41,
# requirements.lock: torch, FlashInfer 0.7.3 with sm100a/sm103a kernels, sgl
# kernels), the environment tested on upstream 314bce7. This image replaces
# that image's SGLang source with this checkout, so staging runs the same
# engine code and dependencies as production, on a model that loads in seconds.
# Shared layers mean GHCR stores only the layers added here.
#
# It meets the confidential-inference chart's aggregated-worker contract
# (helm/confidential-inference/templates/inference-workers.yaml):
#   /usr/local/bin/wait-for-model  verifies the model mount, then supervises
#                                  `python3 -m sglang.launch_server ...`
#   /usr/local/bin/fetch-model     fills a plain model cache (cache.enabled)
#   /usr/local/bin/dcgm-exporter   GPU metrics (--gpu-metrics-port)
#   python3 on PATH                the image's SGLang environment
#   uid 1001, HOME=/tmp            the chart's securityContext
# The three helpers are copied from confidential-inference (vendor/SOURCE).
#
# The HTTP server is hardened as confidential-inference's own SGLang images
# are: harden_http.py --variant pd removes every non-serving route, the OpenAPI
# and docs routes, and every listener other than the Python HTTP frontend. Its
# pinned source hashes are this fork's post-314bce7 files (vendor/SOURCE).
ARG BASE_IMAGE=ghcr.io/confidential-dot-ai/dsv41-pd@sha256:8e1b54fb0aff6d94c0b68c0c3b895841853acbb134d1049675630663d217317e
FROM nvcr.io/nvidia/k8s/dcgm-exporter@sha256:148b0c025e5f2850256816fa33754fbf4733b7a086697a95613fc0ba3db5b003 AS dcgm-exporter

FROM ${BASE_IMAGE}
USER root

# DCGM exporter with its embedded host engine, as confidential-inference's
# images/sglang installs it.
RUN --mount=type=bind,from=dcgm-exporter,source=/,target=/run-dcgm,readonly \
    --mount=type=bind,source=recipes/qwen35-0.8b/container/vendor/dcgm-counters.csv,target=/run-src/dcgm-counters.csv,readonly \
    install -D -m 0555 /run-dcgm/usr/bin/dcgm-exporter /usr/local/bin/dcgm-exporter \
    && install -d -m 0755 /opt/dcgm/lib \
    && cp -a /run-dcgm/usr/lib/x86_64-linux-gnu/libdcgm* /run-dcgm/usr/lib/x86_64-linux-gnu/libnvperf_* /opt/dcgm/lib/ \
    && cp -a /run-dcgm/licenses /opt/dcgm/licenses \
    && install -D -m 0644 /run-src/dcgm-counters.csv /etc/dcgm-exporter/counters.csv

RUN --mount=type=bind,source=recipes/qwen35-0.8b/container/vendor,target=/run-src,readonly \
    install -D -m 0555 /run-src/wait_for_model.py /usr/local/bin/wait-for-model \
    && install -D -m 0555 /run-src/fetch_model.py /usr/local/bin/fetch-model

# This checkout's SGLang over the base image's. The environment's editable
# install already points at /opt/sglang/python. The DeepSeek-V4.1 recipe goes:
# this image does not run it.
WORKDIR /opt/sglang
RUN rm -rf /opt/sglang/python /opt/sglang/recipes /opt/sglang/SOURCE_RECORD.json
COPY python python
COPY recipes/qwen35-0.8b recipes/qwen35-0.8b
# Hardened, then byte-compiled (the runtime user cannot write the root-owned
# tree). The imports need no GPU; the GPU boot is the kernel check.
RUN --mount=type=bind,source=recipes/qwen35-0.8b/container/vendor/harden_http.py,target=/run-src/harden-http.py,readonly \
    python -I -B /run-src/harden-http.py --root /opt/sglang/python/sglang --variant pd \
    && python -c "import sglang, sglang.srt.models.qwen3_5, sglang.launch_server, sglang.srt.entrypoints.http_server" \
    && python -m compileall -q python/sglang > /dev/null \
    && chmod -R go-w /opt/sglang

ARG SOURCE_RECORD
RUN test -n "${SOURCE_RECORD}" && printf '%s\n' "${SOURCE_RECORD}" > /opt/sglang/SOURCE_RECORD.json \
    && python -c "import json; json.load(open('/opt/sglang/SOURCE_RECORD.json'))"

LABEL org.opencontainers.image.title="confidential-sglang Qwen3.5-0.8B (1 GPU, staging)"
LABEL org.opencontainers.image.source="https://github.com/confidential-dot-ai/confidential-sglang"

ENV HOME=/tmp
USER 1001:1001
ENTRYPOINT ["/opt/sglang/recipes/qwen35-0.8b/container/serve.sh"]
CMD ["serve"]
