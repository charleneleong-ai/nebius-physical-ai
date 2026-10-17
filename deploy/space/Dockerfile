# The public demo on a Hugging Face Space (CPU): `npai demo` serving the live trust-gate check and the gallery of
# rehearsed scenes (gallery/, rendered with deploy/gallery.sh). NEBIUS_API_KEY is a Space secret; uploads are capped.
FROM ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 AS uv

FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641
COPY --from=uv /uv /bin/
RUN useradd -m -u 1000 user
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python GALLERY_DIR=/app/gallery
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache
COPY src ./src
RUN uv sync --frozen --no-dev --no-cache
COPY gallery ./gallery
USER user
EXPOSE 7860
CMD ["/app/.venv/bin/npai", "demo", "--port", "7860"]
