module "network" {
  source = "./modules/network"

  project_name        = var.project_name
  environment         = var.environment
  vpc_cidr            = var.vpc_cidr
  public_subnet_cidrs = var.public_subnet_cidrs
  availability_zones  = var.availability_zones
  container_port      = var.container_port
  tags                = var.tags
}

module "service" {
  source = "./modules/service"

  project_name         = var.project_name
  environment          = var.environment
  aws_region           = var.aws_region
  vpc_id               = module.network.vpc_id
  public_subnet_ids    = module.network.public_subnet_ids
  alb_security_group   = module.network.alb_security_group_id
  tasks_security_group = module.network.tasks_security_group_id
  container_image      = var.container_image
  container_port       = var.container_port
  cpu                  = var.cpu
  memory               = var.memory
  desired_count        = var.desired_count
  health_check_path    = var.health_check_path
  log_retention_days   = var.log_retention_days
  tags                 = var.tags
}
