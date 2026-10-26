import os

# Model & Index configurations
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-m3")
# Backend can be "auto", "ollama", "keyword", "onnx", "pytorch", or "mock".
# "auto" uses Ollama's bge-m3 (same model the index was built with) when it is
# reachable, otherwise falls back to TF-IDF keyword search over the transcripts.
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "auto")
INDEX_PATH = os.getenv("INDEX_PATH", "embeddings.joblib")

# Ollama (embeddings + answer generation)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
# "auto" uses Ollama for answers when reachable, "none" always uses the extractive answer
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto")

# User-uploaded courses
DATA_DIR = os.getenv("DATA_DIR", "data")
# faster-whisper model size: tiny, base, small, medium, large-v2, large-v3 (bigger = slower, more accurate)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))

# SLA thresholds
SLA_LATENCY_MS = float(os.getenv("SLA_LATENCY_MS", "300.0"))

# Cost tracking configurations (AWS Pricing)
# AWS Lambda Pricing: 
# Duration cost per millisecond for arm64 architecture with different memory limits:
# 1024 MB: $0.0000000133 per ms
# 2048 MB: $0.0000000267 per ms
# 4096 MB: $0.0000000533 per ms
AWS_LAMBDA_MEM_MB = int(os.getenv("AWS_LAMBDA_MEM_MB", "2048"))
AWS_LAMBDA_COST_PER_MS = float(os.getenv("AWS_LAMBDA_COST_PER_MS", "0.0000000267"))

# AWS ECS Fargate Pricing:
# vCPU per hour: $0.04048
# GB Memory per hour: $0.004445
# A typical task configuration has 2 vCPUs and 4 GB RAM.
AWS_ECS_VCPU_COUNT = int(os.getenv("AWS_ECS_VCPU_COUNT", "2"))
AWS_ECS_MEM_GB = int(os.getenv("AWS_ECS_MEM_GB", "4"))
AWS_ECS_CPU_COST_PER_SEC = (AWS_ECS_VCPU_COUNT * 0.04048) / 3600.0
AWS_ECS_MEM_COST_PER_SEC = (AWS_ECS_MEM_GB * 0.004445) / 3600.0
AWS_ECS_TASK_COST_PER_SEC = AWS_ECS_CPU_COST_PER_SEC + AWS_ECS_MEM_COST_PER_SEC
