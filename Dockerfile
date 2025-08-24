# GPU-ready Dockerfile for this project
# Assumes CUDA 12.1 on the host (nvidia container runtime)

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Basic system deps and python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl build-essential git python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3 -> python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 || true

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy project
COPY . /app

# Create a requirements file excluding torch and faiss (we will install GPU wheels separately)
RUN python - <<'PY'
from pathlib import Path
r = Path('requirements.txt').read_text()
lines = [l for l in r.splitlines() if l.strip() and not l.strip().lower().startswith('torch') and 'faiss' not in l.lower()]
Path('req_no_torch_faiss.txt').write_text('\n'.join(lines))
print('Created req_no_torch_faiss.txt')
PY

# Install the rest of python requirements (excluding torch/faiss)
RUN pip install -r req_no_torch_faiss.txt

# Install PyTorch + torchvision for CUDA 12.1 (exact wheel index)
# Note: using the official PyTorch extra index for CUDA 12.1
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.8.0+cu121" "torchvision==0.23.0+cu121"

# FAISS GPU: recommended to install via conda/mamba for GPU support; shown as optional below.
# If you want a pip-based CPU fallback, the following will install the CPU wheel:
# RUN pip install faiss-cpu==1.12.0

# Expose streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
