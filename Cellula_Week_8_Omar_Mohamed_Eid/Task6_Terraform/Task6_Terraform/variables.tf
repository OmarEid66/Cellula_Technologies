variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-west-1"

  validation {
    condition     = can(regex("^[a-z]{2}-[a-z]+-[0-9]$", var.aws_region))
    error_message = "aws_region must look like eu-west-1."
  }
}

variable "project_name" {
  description = "Short project slug used as a prefix for every resource name."
  type        = string
  default     = "cellula-asr"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,24}$", var.project_name))
    error_message = "project_name must be 3-25 lowercase letters, digits or dashes."
  }
}

variable "environment" {
  description = "Deployment stage."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "vpc_cidr" {
  description = "CIDR block of the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for the two public subnets (one per AZ)."
  type        = list(string)
  default     = ["10.20.1.0/24", "10.20.2.0/24"]

  validation {
    condition     = length(var.public_subnet_cidrs) == 2
    error_message = "Exactly two public subnet CIDRs are required."
  }
}

variable "availability_zones" {
  description = "Availability zones that match public_subnet_cidrs, in order."
  type        = list(string)
  default     = ["eu-west-1a", "eu-west-1b"]

  validation {
    condition     = length(var.availability_zones) == length(var.public_subnet_cidrs)
    error_message = "availability_zones must have one entry per public subnet."
  }
}

variable "container_image" {
  description = "Fully qualified image (ECR or Docker Hub) of the FastAPI service."
  type        = string
  default     = "public.ecr.aws/nginx/nginx:1.27"
}

variable "container_port" {
  description = "Port the container listens on."
  type        = number
  default     = 8000

  validation {
    condition     = var.container_port > 0 && var.container_port < 65536
    error_message = "container_port must be a valid TCP port."
  }
}

variable "cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)."
  type        = number
  default     = 1024

  validation {
    condition     = contains([256, 512, 1024, 2048, 4096], var.cpu)
    error_message = "cpu must be one of 256, 512, 1024, 2048, 4096."
  }
}

variable "memory" {
  description = "Fargate task memory in MiB; must follow the CPU/memory matrix."
  type        = number
  default     = 2048

  validation {
    condition     = var.memory >= 512 && var.memory <= 30720
    error_message = "memory must be between 512 and 30720 MiB."
  }
}

variable "desired_count" {
  description = "Number of running tasks behind the load balancer."
  type        = number
  default     = 2

  validation {
    condition     = var.desired_count >= 1 && var.desired_count <= 10
    error_message = "desired_count must be between 1 and 10."
  }
}

variable "health_check_path" {
  description = "HTTP path used by the ALB target group health check."
  type        = string
  default     = "/health"
}

variable "log_retention_days" {
  description = "CloudWatch retention for container logs."
  type        = number
  default     = 14

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90], var.log_retention_days)
    error_message = "log_retention_days must be an allowed CloudWatch retention value."
  }
}

variable "tags" {
  description = "Extra tags merged into the module defaults."
  type        = map(string)
  default     = {}
}
