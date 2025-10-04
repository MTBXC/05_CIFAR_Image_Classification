# Outputs for CIFAR-10 CNN AWS infrastructure

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "URL of the API service"
  value       = "http://${aws_lb.main.dns_name}"
}

output "mlflow_url" {
  description = "URL of the MLflow service"
  value       = "http://${aws_lb.main.dns_name}/mlflow"
}

output "ecr_api_repository_url" {
  description = "URL of the API ECR repository"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_mlflow_repository_url" {
  description = "URL of the MLflow ECR repository"
  value       = aws_ecr_repository.mlflow.repository_url
}

output "ecr_training_repository_url" {
  description = "URL of the Training ECR repository"
  value       = aws_ecr_repository.training.repository_url
}

output "s3_mlflow_artifacts_bucket" {
  description = "Name of the MLflow artifacts S3 bucket"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "s3_model_storage_bucket" {
  description = "Name of the model storage S3 bucket"
  value       = aws_s3_bucket.model_storage.bucket
}

output "s3_training_data_bucket" {
  description = "Name of the training data S3 bucket"
  value       = aws_s3_bucket.training_data.bucket
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.mlflow.endpoint
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.mlflow.port
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

