variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS Region to deploy resource"
}

variable "app_name" {
  type        = string
  default     = "sigma-rag-serving"
  description = "Application name for tagging and naming"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Target environment name"
}

variable "ecs_cpu" {
  type        = number
  default     = 2048
  description = "CPU units for ECS Task (1024 = 1 vCPU)"
}

variable "ecs_memory" {
  type        = number
  default     = 4096
  description = "Memory (in MB) for ECS Task"
}

variable "ecs_desired_count" {
  type        = number
  default     = 3
  description = "Initial replicas of ECS Task running"
}

variable "container_port" {
  type        = number
  default     = 8000
  description = "The port exposed by the FastAPI container"
}

variable "lambda_memory_size" {
  type        = number
  default     = 2048
  description = "Memory size allocated for AWS Lambda in MB"
}
