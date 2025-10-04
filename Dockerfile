# Use official TensorFlow image as base (much faster than installing from scratch)
FROM tensorflow/tensorflow:2.20.0-jupyter

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AWS_DEFAULT_REGION=us-east-1

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching (without TensorFlow - already in base image)
COPY requirements-docker.txt .

# Install Python dependencies (cached layer)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Create necessary directories
RUN mkdir -p /app/data/raw /app/models /app/mlruns /app/web/public

# Download CIFAR-10 data (cached layer)
RUN cd /app/data/raw && \
    wget -q https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && \
    tar -xzf cifar-10-python.tar.gz && \
    rm cifar-10-python.tar.gz

# Copy only necessary source code (this layer will change most often)
COPY src/ /app/src/
COPY api/ /app/api/
COPY web/ /app/web/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the application
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]