FROM python:3.10
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
WORKDIR /app
# Update system, install git, curl and zstd
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*
# Install Ollama CLI
RUN curl -fsSL https://ollama.com/install.sh | sh
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "docker/docker_script.py"]