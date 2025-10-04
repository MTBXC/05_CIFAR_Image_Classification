# SageMaker infrastructure for CIFAR-10 model training with MLFlow integration
# This extends the existing MLFlow infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Get the latest Ubuntu 22.04 LTS AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Local values
locals {
  project_name = "cifar-sagemaker"
  environment  = var.environment
  
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "terraform"
  }
}

# VPC for SageMaker (can reuse existing VPC if needed)
resource "aws_vpc" "sagemaker_vpc" {
  cidr_block           = var.sagemaker_vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-vpc"
  })
}

# Internet Gateway
resource "aws_internet_gateway" "sagemaker_igw" {
  vpc_id = aws_vpc.sagemaker_vpc.id

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-igw"
  })
}

# Public Subnets (SageMaker requires at least 2 AZs)
resource "aws_subnet" "sagemaker_public_1" {
  vpc_id                  = aws_vpc.sagemaker_vpc.id
  cidr_block              = var.sagemaker_public_subnet_1_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-public-subnet-1"
  })
}

resource "aws_subnet" "sagemaker_public_2" {
  vpc_id                  = aws_vpc.sagemaker_vpc.id
  cidr_block              = var.sagemaker_public_subnet_2_cidr
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-public-subnet-2"
  })
}

# Private Subnets for SageMaker training jobs
resource "aws_subnet" "sagemaker_private_1" {
  vpc_id            = aws_vpc.sagemaker_vpc.id
  cidr_block        = var.sagemaker_private_subnet_1_cidr
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-private-subnet-1"
  })
}

resource "aws_subnet" "sagemaker_private_2" {
  vpc_id            = aws_vpc.sagemaker_vpc.id
  cidr_block        = var.sagemaker_private_subnet_2_cidr
  availability_zone = data.aws_availability_zones.available.names[1]

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-private-subnet-2"
  })
}

# Route Tables
resource "aws_route_table" "sagemaker_public" {
  vpc_id = aws_vpc.sagemaker_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.sagemaker_igw.id
  }

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-public-rt"
  })
}

resource "aws_route_table" "sagemaker_private" {
  vpc_id = aws_vpc.sagemaker_vpc.id

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-private-rt"
  })
}

# Route Table Associations
resource "aws_route_table_association" "sagemaker_public_1" {
  subnet_id      = aws_subnet.sagemaker_public_1.id
  route_table_id = aws_route_table.sagemaker_public.id
}

resource "aws_route_table_association" "sagemaker_public_2" {
  subnet_id      = aws_subnet.sagemaker_public_2.id
  route_table_id = aws_route_table.sagemaker_public.id
}

resource "aws_route_table_association" "sagemaker_private_1" {
  subnet_id      = aws_subnet.sagemaker_private_1.id
  route_table_id = aws_route_table.sagemaker_private.id
}

resource "aws_route_table_association" "sagemaker_private_2" {
  subnet_id      = aws_subnet.sagemaker_private_2.id
  route_table_id = aws_route_table.sagemaker_private.id
}

# NAT Gateway for private subnets
resource "aws_eip" "sagemaker_nat" {
  domain = "vpc"

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-nat-eip"
  })
}

resource "aws_nat_gateway" "sagemaker_nat" {
  allocation_id = aws_eip.sagemaker_nat.id
  subnet_id     = aws_subnet.sagemaker_public_1.id

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-nat-gateway"
  })

  depends_on = [aws_internet_gateway.sagemaker_igw]
}

# Add route to NAT Gateway for private subnets
resource "aws_route" "sagemaker_private_nat" {
  route_table_id         = aws_route_table.sagemaker_private.id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.sagemaker_nat.id
}

# S3 Bucket for SageMaker data and models
resource "aws_s3_bucket" "sagemaker_data" {
  bucket = "${local.project_name}-data-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-data-bucket"
  })
}

resource "aws_s3_bucket" "sagemaker_models" {
  bucket = "${local.project_name}-models-${random_string.bucket_suffix.result}"

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-models-bucket"
  })
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket configurations
resource "aws_s3_bucket_versioning" "sagemaker_data" {
  bucket = aws_s3_bucket.sagemaker_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "sagemaker_models" {
  bucket = aws_s3_bucket.sagemaker_models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "sagemaker_data" {
  bucket = aws_s3_bucket.sagemaker_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "sagemaker_models" {
  bucket = aws_s3_bucket.sagemaker_models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Role for SageMaker Execution
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${local.project_name}-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for SageMaker Execution Role
resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name = "${local.project_name}-execution-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.sagemaker_data.arn,
          "${aws_s3_bucket.sagemaker_data.arn}/*",
          aws_s3_bucket.sagemaker_models.arn,
          "${aws_s3_bucket.sagemaker_models.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      }
    ]
  })
}

# IAM Role for SageMaker Notebook Instance
resource "aws_iam_role" "sagemaker_notebook_role" {
  name = "${local.project_name}-notebook-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

# IAM Policy for SageMaker Notebook Role
resource "aws_iam_role_policy" "sagemaker_notebook_policy" {
  name = "${local.project_name}-notebook-policy"
  role = aws_iam_role.sagemaker_notebook_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.sagemaker_data.arn,
          "${aws_s3_bucket.sagemaker_data.arn}/*",
          aws_s3_bucket.sagemaker_models.arn,
          "${aws_s3_bucket.sagemaker_models.arn}/*",
          "arn:aws:s3:::mlflow-minimal-*",
          "arn:aws:s3:::mlflow-minimal-*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "main" {
  name          = "${local.project_name}-notebook"
  role_arn      = aws_iam_role.sagemaker_notebook_role.arn
  instance_type = var.notebook_instance_type

  # Security configuration
  subnet_id = aws_subnet.sagemaker_private_1.id
  security_groups = [aws_security_group.sagemaker_notebook.id]

  # Root access
  root_access = "Enabled"

  # Volume size
  volume_size = var.notebook_volume_size

  tags = local.common_tags
}

# SageMaker Notebook Instance Lifecycle Configuration - disabled for now
# resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "main" {
#   name = "${local.project_name}-lifecycle-config"
#
#   on_create = base64encode(templatefile("${path.module}/notebook_lifecycle.sh", {
#     mlflow_tracking_uri = var.mlflow_tracking_uri
#   }))
#
#   on_start = base64encode(templatefile("${path.module}/notebook_start.sh", {
#     mlflow_tracking_uri = var.mlflow_tracking_uri
#   }))
# }

# Security Group for SageMaker Notebook
resource "aws_security_group" "sagemaker_notebook" {
  name_prefix = "${local.project_name}-notebook-"
  vpc_id      = aws_vpc.sagemaker_vpc.id

  # HTTPS access
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-notebook-sg"
  })
}

# Security Group for SageMaker Training Jobs
resource "aws_security_group" "sagemaker_training" {
  name_prefix = "${local.project_name}-training-"
  vpc_id      = aws_vpc.sagemaker_vpc.id

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-training-sg"
  })
}

# VPC Endpoint for S3 (to avoid NAT Gateway costs for S3 access)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.sagemaker_vpc.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-s3-endpoint"
  })
}

# VPC Endpoint Route Table Association
resource "aws_vpc_endpoint_route_table_association" "s3_private" {
  vpc_endpoint_id = aws_vpc_endpoint.s3.id
  route_table_id  = aws_route_table.sagemaker_private.id
}
