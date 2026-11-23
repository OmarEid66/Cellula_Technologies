output "vpc_id" {
  description = "ID of the created VPC."
  value       = module.network.vpc_id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets used by the load balancer and tasks."
  value       = module.network.public_subnet_ids
}

output "alb_dns_name" {
  description = "Public DNS name of the application load balancer."
  value       = module.service.alb_dns_name
}

output "service_url" {
  description = "Convenience URL for the running API."
  value       = "http://${module.service.alb_dns_name}"
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster hosting the service."
  value       = module.service.ecs_cluster_name
}

output "log_group_name" {
  description = "CloudWatch log group receiving container logs."
  value       = module.service.log_group_name
}
