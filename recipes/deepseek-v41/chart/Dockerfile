# syntax=docker/dockerfile:1
# DeepSeek-V4.1-Flash worker image for the confidential-inference chart.
# Build with build.sh; the build context is the repository root.
#
# The Python environment is the PD recipe image's (../container: the
# requirements.lock environment tested on upstream 314bce7, FlashInfer 0.7.3
# with sm100a/sm103a kernels). This image replaces that image's SGLang and
# recipes/deepseek-v41 with this checkout, so it runs the same engine code as
# the docker recipe and as recipes/qwen35-0.8b (the staging worker).
#
# It meets both worker contracts of the chart
# (helm/confidential-inference/templates/):
#   inference-workers.yaml (aggregated, inference.pd.enabled=false)
#     /usr/local/bin/wait-for-model   verifies the model, then supervises
#                                     `python3 -m sglang.launch_server ...`
#     /usr/local/bin/fetch-model      inference.model.download
#     /usr/local/bin/dcgm-exporter    --gpu-metrics-port
#   inference-pd.yaml (inference.pd.enabled=true)
#     /opt/dsv41/bin/python           inference.pd.python
#     /opt/sglang/recipes/deepseek-v41/launcher/hold_engram_cache.py
#                                     inference.pd.engramHolder.script
#     /bin/sh, curl                   the workers' start gate
#   inference-pairs.yaml (inference.pd.paired.enabled=true)
#     /usr/local/bin/sglang-pair      pair supervisor, with pair_diagnostics.py
#                                     and private_log_sink.py beside it
#   all: uid 1001, HOME=/tmp, nothing written under /opt at run time.
#
# The HTTP server is hardened as confidential-inference's own SGLang images
# are: harden_http.py --variant pd removes every non-serving route, the OpenAPI
# and docs routes, and every listener other than the Python HTTP frontend. It
# refuses to run unless http_server.py and launch_server.py match the sources
# it was reviewed against, so a change to either fails this build.
#
# vendor/ holds those helpers, copied unchanged from confidential-inference
# (vendor/SOURCE).
ARG BASE_IMAGE=ghcr.io/confidential-dot-ai/dsv41-pd@sha256:8e1b54fb0aff6d94c0b68c0c3b895841853acbb134d1049675630663d217317e
FROM nvcr.io/nvidia/k8s/dcgm-exporter@sha256:148b0c025e5f2850256816fa33754fbf4733b7a086697a95613fc0ba3db5b003 AS dcgm-exporter

FROM ${BASE_IMAGE}
USER root

# DCGM exporter with its embedded host engine, as confidential-inference's
# images/sglang installs it.
RUN --mount=type=bind,from=dcgm-exporter,source=/,target=/run-dcgm,readonly \
    --mount=type=bind,source=recipes/deepseek-v41/chart/vendor/dcgm-counters.csv,target=/run-src/dcgm-counters.csv,readonly \
    install -D -m 0555 /run-dcgm/usr/bin/dcgm-exporter /usr/local/bin/dcgm-exporter \
    && install -d -m 0755 /opt/dcgm/lib \
    && cp -a /run-dcgm/usr/lib/x86_64-linux-gnu/libdcgm* /run-dcgm/usr/lib/x86_64-linux-gnu/libnvperf_* /opt/dcgm/lib/ \
    && cp -a /run-dcgm/licenses /opt/dcgm/licenses \
    && install -D -m 0644 /run-src/dcgm-counters.csv /etc/dcgm-exporter/counters.csv

# The chart's helpers, at the paths the chart runs them by.
RUN --mount=type=bind,source=recipes/deepseek-v41/chart/vendor,target=/run-src,readonly \
    install -D -m 0555 /run-src/wait_for_model.py /usr/local/bin/wait-for-model \
    && install -D -m 0555 /run-src/fetch_model.py /usr/local/bin/fetch-model \
    && install -D -m 0555 /run-src/pair_supervisor.py /usr/local/bin/sglang-pair \
    && install -D -m 0444 /run-src/pair_diagnostics.py /usr/local/bin/pair_diagnostics.py \
    && install -D -m 0444 /run-src/private_log_sink.py /usr/local/bin/private_log_sink.py \
    && python -I -B /usr/local/bin/pair_diagnostics.py

# This checkout's SGLang and recipe over the base image's. The environment's
# editable install already points at /opt/sglang/python.
WORKDIR /opt/sglang
RUN rm -rf /opt/sglang/python /opt/sglang/recipes /opt/sglang/SOURCE_RECORD.json
COPY python python
COPY recipes/deepseek-v41 recipes/deepseek-v41
# Harden, then byte-compile (the runtime user cannot write the root-owned
# tree). The imports need no GPU; the GPU boot is the kernel check.
RUN --mount=type=bind,source=recipes/deepseek-v41/chart/vendor/harden_http.py,target=/run-src/harden-http.py,readonly \
    python -I -B /run-src/harden-http.py --root /opt/sglang/python/sglang --variant pd \
    && python -c "import sglang.launch_server, sglang.srt.entrypoints.http_server, sglang.srt.models.deepseek_v4, mooncake" \
    && python -m compileall -q python/sglang recipes/deepseek-v41/launcher > /dev/null \
    && chmod -R go-w /opt/sglang

ARG SOURCE_RECORD
RUN test -n "${SOURCE_RECORD}" && printf '%s\n' "${SOURCE_RECORD}" > /opt/sglang/SOURCE_RECORD.json \
    && python -c "import json; json.load(open('/opt/sglang/SOURCE_RECORD.json'))"

LABEL org.opencontainers.image.title="confidential-sglang DeepSeek-V4.1-Flash (confidential-inference worker)"
LABEL org.opencontainers.image.source="https://github.com/confidential-dot-ai/confidential-sglang"

# The chart sets every container's command; the base image's root
# entrypoint (the docker recipe) does not apply here.
ENV HOME=/tmp
USER 1001:1001
ENTRYPOINT []
CMD ["/opt/dsv41/bin/python", "-m", "sglang.launch_server", "--help"]
