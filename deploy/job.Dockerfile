# Preflight batch job: one phone frame in, the verdict and its rollout videos out (src/nebius_physical_ai/job.py).
# Runs on a Nebius Serverless Job with one GPU; the GR00T checkpoint is mounted at $CKPT (Object Storage volume), never
# baked in. Two environments, as on the A100: the project (Python 3.13: perception, MuJoCo twin, verdict, Token Factory
# calls) and LeRobot 0.6.1 (Python 3.12: the GR00T N1.7 policy server; deploy/lerobot/uv.lock, the A100's working set).
#   docker build -f deploy/job.Dockerfile -t preflight-job .
#   docker run --gpus all -v CKPT_DIR:/models/policy -v $PWD/in:/in -v $PWD/out:/out -e NEBIUS_API_KEY preflight-job \
#     /in/frame.jpg /out --n 32
FROM ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv

# The SO-101 twin model, from the menagerie commit the benchmarks ran on; git stays out of the final image
FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641 AS menagerie
ARG MENAGERIE_COMMIT=4d038b3feae26ec82b46a4d586379114012a8ac7
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && git init -q /opt/menagerie && cd /opt/menagerie \
    && git remote add origin https://github.com/google-deepmind/mujoco_menagerie.git \
    && git sparse-checkout set robotstudio_so101 && git fetch -q --depth 1 origin "$MENAGERIE_COMMIT" \
    && git checkout -q FETCH_HEAD

FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
COPY --from=uv /uv /uvx /bin/
# EGL for headless MuJoCo rendering on the GPU; the NVIDIA runtime injects the driver's EGL when graphics is enabled
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglvnd0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=menagerie /opt/menagerie/robotstudio_so101 /opt/menagerie/robotstudio_so101
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics CKPT=/models/policy \
    PREFLIGHT_SO101_XML=/opt/menagerie/robotstudio_so101/so101.xml
WORKDIR /app

COPY deploy/lerobot/pyproject.toml deploy/lerobot/uv.lock /opt/lerobot/
RUN UV_PROJECT_ENVIRONMENT=/opt/lerobot/.venv uv sync --frozen --no-cache --project /opt/lerobot

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache
COPY src ./src
COPY scripts/lerobot_policy_server.py ./scripts/
RUN uv sync --frozen --no-dev --no-cache
COPY deploy/job_entrypoint.sh ./
ENTRYPOINT ["/app/job_entrypoint.sh"]
