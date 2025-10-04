# Variables for SageMaker infrastructure

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-north-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

# VPC Configuration
variable "sagemaker_vpc_cidr" {
  description = "CIDR block for SageMaker VPC"
  type        = string
  default     = "10.1.0.0/16"
}

variable "sagemaker_public_subnet_1_cidr" {
  description = "CIDR block for first public subnet"
  type        = string
  default     = "10.1.1.0/24"
}

variable "sagemaker_public_subnet_2_cidr" {
  description = "CIDR block for second public subnet"
  type        = string
  default     = "10.1.2.0/24"
}

variable "sagemaker_private_subnet_1_cidr" {
  description = "CIDR block for first private subnet"
  type        = string
  default     = "10.1.10.0/24"
}

variable "sagemaker_private_subnet_2_cidr" {
  description = "CIDR block for second private subnet"
  type        = string
  default     = "10.1.20.0/24"
}

# SageMaker Configuration
variable "notebook_instance_type" {
  description = "SageMaker Notebook instance type"
  type        = string
  default     = "ml.t3.medium"
  validation {
    condition = can(regex("^ml\\.", var.notebook_instance_type))
    error_message = "Notebook instance type must start with 'ml.'"
  }
}

variable "notebook_volume_size" {
  description = "Size of the notebook instance volume in GB"
  type        = number
  default     = 20
  validation {
    condition     = var.notebook_volume_size >= 5 && var.notebook_volume_size <= 16384
    error_message = "Volume size must be between 5 and 16384 GB."
  }
}

# MLFlow Integration
variable "mlflow_tracking_uri" {
  description = "MLFlow tracking URI for integration"
  type        = string
  default     = "http://localhost:5000"
}

# Training Configuration
variable "training_instance_type" {
  description = "SageMaker training instance type"
  type        = string
  default     = "ml.m5.large"
  validation {
    condition = can(regex("^ml\\.", var.training_instance_type))
    error_message = "Training instance type must start with 'ml.'"
  }
}

variable "training_instance_count" {
  description = "Number of training instances"
  type        = number
  default     = 1
  validation {
    condition     = var.training_instance_count >= 1 && var.training_instance_count <= 10
    error_message = "Training instance count must be between 1 and 10."
  }
}

# Model Configuration
variable "model_name" {
  description = "Name of the model to train"
  type        = string
  default     = "cifar10-cnn"
}

variable "model_version" {
  description = "Version of the model"
  type        = string
  default     = "1.0"
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}












