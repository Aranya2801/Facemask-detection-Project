# FaceMask Detection System v2.0 — Dockerfile
FROM python:3.9-slim

LABEL maintainer="Aranya2801"
LABEL description="FaceMask Detection System v2.0"
LABEL version="2.0.0"

# System dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create dirs
RUN mkdir -p logs/screenshots models dataset

# Expose port
EXPOSE 5000

# Default: web server
CMD ["python", "web/app.py", "--host", "0.0.0.0", "--port", "5000"]
