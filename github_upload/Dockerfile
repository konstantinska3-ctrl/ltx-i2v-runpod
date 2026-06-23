FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt /app/requirements-server.txt
RUN pip install --no-cache-dir -r /app/requirements-server.txt

COPY server.py /app/server.py

ENV HF_HOME=/workspace/hf-cache
ENV LTX_MODEL_ID=Lightricks/LTX-Video
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
