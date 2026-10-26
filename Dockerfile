# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.9-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a wheels directory to avoid compiling in final image
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.9-slim AS runtime

# Set environment variables for Python optimization and stability
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    MODEL_BACKEND="auto"

# Create a non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local

# Copy application files
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup embeddings.joblib .

# Set permissions; data/ holds uploaded videos and transcripts
RUN mkdir -p /app/data && chown appuser:appgroup /app/data && chmod -R 755 /app

# Switch to the non-root user
USER appuser

EXPOSE 8000

# Run uvicorn with 4 workers for multi-core performance
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
