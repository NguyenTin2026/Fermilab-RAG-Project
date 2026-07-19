# Fermilab RAG Q&A Assistant — container cho Streamlit app.
#
# LUU Y QUAN TRONG:
#   - Image nay KHONG chua weights model (ft-qwen3-fermilab/) va KHONG chua
#     corpus LFS. Chung qua lon / nam ngoai git. Ban phai cung cap luc chay:
#       * Mount thu muc weights:   -v /duong/dan/ft-qwen3-fermilab:/app/Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab
#       * Va corpus_st7.jsonl (git lfs pull trên host roi mount, hoac COPY vao).
#   - Muon chay GPU: dung base image CUDA + `docker run --gpus all`.
#     Base image duoi day la CPU (chay duoc nhung inference RAT cham).
#
# Build:  docker build -t fermilab-rag .
# Run:    docker run -p 8501:8501 --env-file .env \
#             -v $PWD/Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab:/app/Fermilab-process_data_and_finetune_model/ft-qwen3-fermilab \
#             fermilab-rag

FROM python:3.12-slim

# Thu vien he thong toi thieu (PyMuPDF, tokenizers... can build tools).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cai dependencies truoc (tan dung layer cache).
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code (crawled_data/ va weights bi loai qua .dockerignore de image nhe).
COPY . .

EXPOSE 8501

# Chay tren compute node offline: HF khong goi mang.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

WORKDIR /app/Fermilab-process_data_and_finetune_model
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]
