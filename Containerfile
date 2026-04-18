FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04 as cuda_builder
FROM ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm

WORKDIR /nonlocal-matchfilter

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

COPY --from=cuda_builder /usr/local/cuda/ /usr/local/cuda/
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="5.0 5.2 5.2+PTX 5.3 6.0 6.1 6.1+PTX 6.2 7.0 7.0+PTX 7.2 7.5 7.5+PTX 8.0 8.6 8.6+PTX 8.7 8.9 8.9+PTX 9.0 9.0+PTX 9.0a"

ENV HYDRA_FULL_ERROR=1
ENV AIM_REPO_PATH=/aim
ENV PNG_DATASETS_PATH=/data/png
ENV RAW_DATASETS_PATH=/data/raw

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /nonlocal-matchfilter
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV PATH="/nonlocal-matchfilter/.venv/bin:$PATH"
