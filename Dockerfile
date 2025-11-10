
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# --------------------------
# Metadata
# --------------------------
LABEL description="WAN T2V RP Handler - RunPod Serverless"

# --------------------------
# Environment variables
# --------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# --------------------------
# System dependencies
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
# Create working directory
# --------------------------
WORKDIR /workspace

# --------------------------
# Clone repository
# --------------------------
RUN git clone https://github.com/ModelTC/Wan2.2-Lightning.git

# --------------------------
# Copy your handler into repo
# --------------------------
COPY rp_handler.py /workspace/Wan2.2-Lightning/rp_handler.py

# --------------------------
# Python dependencies
# --------------------------
WORKDIR /workspace/Wan2.2-Lightning
RUN pip install runpod librosa decord hf_transfer
RUN pip install flash_attn --no-build-isolation
RUN pip install -r requirements.txt
RUN pip install google-cloud-storage
# --------------------------
# GPU environment (optional)
# --------------------------
ENV CUDA_VISIBLE_DEVICES=0

# --------------------------
# Serverless entrypoint
# --------------------------
CMD ["python", "-u", "rp_handler.py"]
