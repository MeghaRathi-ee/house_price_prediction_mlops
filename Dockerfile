# ─────────────────────────────────────────────
# Base image
# python:3.10-slim = python 3.10 + minimal OS
# slim = smaller image size (no unnecessary tools)
# Never use :latest tag — always pin version
# so image builds are reproducible
# ─────────────────────────────────────────────
FROM python:3.10-slim

# ─────────────────────────────────────────────
# Set working directory inside container
# All subsequent commands run from here
# ─────────────────────────────────────────────
WORKDIR /app

# ─────────────────────────────────────────────
# Install system dependencies
# gcc: needed for some Python packages to compile
# --no-install-recommends: keeps image small
# rm -rf /var/lib/apt/lists/*: cleans apt cache
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gcc \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Copy requirements first — before code
# Why? Docker layer caching.
# If only code changes (not requirements),
# Docker skips the pip install step entirely
# This makes rebuilds much faster
# ─────────────────────────────────────────────
COPY requirements.txt .

# ─────────────────────────────────────────────
# Install Python dependencies
# --no-cache-dir: don't cache pip downloads
# keeps image size smaller
# ─────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.104.0 \
        uvicorn==0.24.0 \
        pandas==2.1.0 \
        numpy==1.24.0 \
        scikit-learn==1.3.0 \
        pyyaml==6.0.1 \
        xgboost==2.0.0 \
        lightgbm==4.1.0

# ─────────────────────────────────────────────
# Copy application code
# Done AFTER pip install so code changes
# don't invalidate the pip cache layer
# ─────────────────────────────────────────────
COPY app/ ./app/
COPY src/ ./src/
COPY params.yaml .

# ─────────────────────────────────────────────
# Copy model artifacts
# These are needed at runtime for predictions
# In production you'd pull these from S3/Azure
# instead of baking them into the image
# ─────────────────────────────────────────────
COPY model.pkl .
COPY data/processed/preprocessor.pkl ./data/processed/preprocessor.pkl
COPY data/processed/outlier_bounds.json ./data/processed/outlier_bounds.json
COPY data/processed/selected_features.json ./data/processed/selected_features.json

# ─────────────────────────────────────────────
# Expose port 8000
# This documents which port the container uses
# Doesn't actually publish it — that's done
# with -p flag when running the container
# ─────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────
# Health check
# Docker pings /health every 30 seconds
# If it fails 3 times → container marked unhealthy
# ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ─────────────────────────────────────────────
# Start command
# host 0.0.0.0 = listen on all network interfaces
# (not just localhost) so requests from outside
# the container can reach it
# workers 1 = single worker (add more for production)
# ─────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]