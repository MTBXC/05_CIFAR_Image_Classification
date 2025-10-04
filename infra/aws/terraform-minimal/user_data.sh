#!/bin/bash

# User data script for MLFlow EC2 instance setup
# This script installs Docker and sets up MLFlow server

set -e

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Create application directory
mkdir -p /home/ubuntu/mlflow-server
cd /home/ubuntu/mlflow-server

# Create Dockerfile for MLflow
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /mlflow

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db \
    AWS_DEFAULT_REGION=${aws_region}

RUN apt-get update && apt-get install -y \
    gcc g++ curl sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.15.0 \
    boto3==1.34.0 \
    awscli==1.32.0

RUN mkdir -p /mlflow/mlruns /mlflow/artifacts

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=30s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "s3://${s3_bucket_name}/mlflow-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  mlflow:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://${s3_bucket_name}/mlflow-artifacts
      - AWS_DEFAULT_REGION=${aws_region}
    volumes:
      - mlflow_data:/mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

volumes:
  mlflow_data:
EOF

# Create startup script
cat > start_mlflow.sh << 'EOF'
#!/bin/bash
cd /home/ubuntu/mlflow-server
echo "Starting MLflow server..."
docker-compose up -d
echo "MLflow server started!"
docker-compose ps
EOF

chmod +x start_mlflow.sh

# Create systemd service for auto-start
cat > /etc/systemd/system/mlflow.service << 'EOF'
[Unit]
Description=MLflow Server
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/mlflow-server
ExecStart=/home/ubuntu/mlflow-server/start_mlflow.sh
User=ubuntu
Group=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
systemctl enable mlflow.service

# Set ownership
chown -R ubuntu:ubuntu /home/ubuntu/mlflow-server

# Create a health check script
cat > /home/ubuntu/health_check.sh << 'EOF'
#!/bin/bash
echo "Checking MLflow service..."
echo "MLflow: $(curl -s -o /dev/null -w '%%{http_code}' http://localhost:5000/health || echo 'DOWN')"
echo "Docker containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
EOF

chmod +x /home/ubuntu/health_check.sh

# Start MLflow immediately
echo "Starting MLflow server..."
cd /home/ubuntu/mlflow-server
docker-compose up -d

# Log completion
echo "User data script completed at $(date)" >> /var/log/user-data.log
echo "MLflow server should be running on port 5000" >> /var/log/user-data.log