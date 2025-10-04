# Outputs for minimal MLFlow AWS infrastructure

output "ec2_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_eip.main.public_ip
}

output "ec2_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.main.public_dns
}

output "mlflow_url" {
  description = "URL of the MLflow service"
  value       = "http://${aws_eip.main.public_ip}:5000"
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow.bucket
}

output "ssh_command" {
  description = "SSH command to connect to the EC2 instance"
  value       = "ssh -i ~/.ssh/id_rsa ubuntu@${aws_eip.main.public_ip}"
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI for client connections"
  value       = "http://${aws_eip.main.public_ip}:5000"
}