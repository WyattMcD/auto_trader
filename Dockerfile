FROM ubuntu:latest
LABEL authors="wyattmcdonald"

ENTRYPOINT ["top", "-b"]
# Stage 1: builder — install dependencies
FROM python:3.11-slim as builder

# system deps for building wheels (add more if your requirements need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt

# Stage 2: runtime — smaller final image
FROM python:3.11-slim

RUN useradd -m botuser
WORKDIR /app

# copy requirements and install directly
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/logs && chown -R botuser:botuser /app
USER botuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "auto_trader.py"]
