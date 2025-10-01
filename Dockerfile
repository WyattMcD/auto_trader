# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (Optional) build deps if your wheels need them; remove if not needed
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv that won't be clobbered by the source code mount
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt


# Make log/state dirs and a non-root user
RUN useradd -m botuser && mkdir -p /app/logs /app/state && chown -R botuser:botuser /app
USER botuser

# Default command; in dev we’ll override with a reloader
CMD ["python", "auto_trader.py"]
