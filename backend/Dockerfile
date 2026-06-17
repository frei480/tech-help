FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim as builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# disable python downloads
ENV UV_PYTHON_DOWNLOADS=0

# Copy uv binaries from the official uv image
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
--mount=type=bind,source=uv.lock,target=uv.lock \
--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
uv sync --locked --no-install-project

# Copy project files needed for dependency installation
COPY app ./app
COPY pyproject.toml  uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.14-slim 

COPY ripgrep_15.1.0-1_amd64.deb .
RUN dpkg -i ripgrep_15.1.0-1_amd64.deb

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Copy the application from the builder
COPY --from=builder --chown=nonroot:nonroot /app /app


# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Use the non-root user to run our application
USER nonroot
# Use `/app` as the working directory
WORKDIR /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
