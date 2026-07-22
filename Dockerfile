FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY weights/ ./weights/

# Install dependencies from lockfile (no venv, install to system)
RUN uv sync --frozen --no-dev

# Run uvicorn from the root directory, pointing to the app inside the src folder
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]