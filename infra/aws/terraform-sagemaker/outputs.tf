# Outputs for SageMaker infrastructure

output "sagemaker_vpc_id" {
  description = "ID of the SageMaker VPC"
  value       = aws_vpc.sagemaker_vpc.id
}

output "sagemaker_public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = [aws_subnet.sagemaker_public_1.id, aws_subnet.sagemaker_public_2.id]
}

output "sagemaker_private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = [aws_subnet.sagemaker_private_1.id, aws_subnet.sagemaker_private_2.id]
}

output "sagemaker_data_bucket_name" {
  description = "Name of the S3 bucket for training data"
  value       = aws_s3_bucket.sagemaker_data.bucket
}

output "sagemaker_models_bucket_name" {
  description = "Name of the S3 bucket for trained models"
  value       = aws_s3_bucket.sagemaker_models.bucket
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sagemaker_notebook_role_arn" {
  description = "ARN of the SageMaker notebook role"
  value       = aws_iam_role.sagemaker_notebook_role.arn
}

output "sagemaker_notebook_instance_name" {
  description = "Name of the SageMaker notebook instance"
  value       = aws_sagemaker_notebook_instance.main.name
}

output "sagemaker_notebook_url" {
  description = "URL of the SageMaker notebook instance"
  value       = "https://${var.aws_region}.console.aws.amazon.com/sagemaker/home?region=${var.aws_region}#/notebook-instances/${aws_sagemaker_notebook_instance.main.name}"
}

output "sagemaker_training_security_group_id" {
  description = "ID of the security group for training jobs"
  value       = aws_security_group.sagemaker_training.id
}

output "sagemaker_notebook_security_group_id" {
  description = "ID of the security group for notebook instance"
  value       = aws_security_group.sagemaker_notebook.id
}

output "mlflow_tracking_uri" {
  description = "MLFlow tracking URI for SageMaker integration"
  value       = var.mlflow_tracking_uri
}

output "training_data_s3_uri" {
  description = "S3 URI for training data"
  value       = "s3://${aws_s3_bucket.sagemaker_data.bucket}/data/"
}

output "models_s3_uri" {
  description = "S3 URI for trained models"
  value       = "s3://${aws_s3_bucket.sagemaker_models.bucket}/models/"
}












