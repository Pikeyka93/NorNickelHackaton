# ============================================================
# ВЛАДЕЛЕЦ: P3 (Docker + стабильность деплоя)
# TODO(P3): держать образ рабочим; --no-cache-dir уже включён (PIP_NO_CACHE_DIR=1).
# ============================================================
# CUDA 12.2 runtime to match the L4 server (CUDA 12.2). Build the image once, run
# API and UI from it. IMPORTANT: run with `--gpus all` so the container sees the GPU.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: python + OpenCV runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install torch built for CUDA 12.1 (compatible with the 12.2 driver), then the rest.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

COPY . .

# GPU visibility check — fails loudly at container start if --gpus was forgotten.
COPY scripts/check_gpu.py /app/scripts/check_gpu.py

EXPOSE 8000 8501

# Default: run API + UI together. Override the CMD to run just one, or training.
#   docker run --gpus all -p 8000:8000 -p 8501:8501 <img>
#   docker run --gpus all <img> python scripts/train.py
CMD ["bash", "-lc", "python scripts/check_gpu.py; \
     uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
     streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501"]
