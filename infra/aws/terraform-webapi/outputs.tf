output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.webapi.repository_url
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "api_url" {
  description = "Public URL of the API"
  value       = "http://${aws_lb.main.dns_name}"
}

output "api_health_check_url" {
  description = "Health check URL"
  value       = "http://${aws_lb.main.dns_name}/api/health"
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.webapi.name
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = [aws_subnet.private_1.id, aws_subnet.private_2.id]
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = [aws_subnet.public_1.id, aws_subnet.public_2.id]
}

output "security_group_ecs_tasks" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "security_group_alb" {
  description = "Security group ID for ALB"
  value       = aws_security_group.alb.id
}

# Deployment information
output "deployment_info" {
  description = "Deployment information summary"
  value = {
    api_url                = "http://${aws_lb.main.dns_name}"
    health_check_url      = "http://${aws_lb.main.dns_name}/api/health"
    ecr_repository        = aws_ecr_repository.webapi.repository_url
    ecs_cluster           = aws_ecs_cluster.main.name
    ecs_service           = aws_ecs_service.webapi.name
    region                = var.aws_region
    environment           = var.environment
  }
}











