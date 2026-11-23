# Task 6 — Terraform for the Week 8 API

Infrastructure as code for the FastAPI service built in Task 5: a public
Application Load Balancer in front of an ECS Fargate service, in its own VPC,
with logs and IAM roles. Two modules keep the root thin: `network` owns the
VPC, subnets, route table and security groups; `service` owns the load
balancer, cluster, task definition and ECS service.

## Layout

| File | Contents |
| --- | --- |
| `providers.tf` | Terraform and AWS provider pins, default tags |
| `main.tf` | Two module calls, nothing else |
| `variables.tf` | Every root input, typed and validated |
| `outputs.tf` | VPC, subnets, ALB DNS, service URL, cluster and log group |
| `terraform.tfvars.example` | Example values; copy to `terraform.tfvars` |
| `modules/network/` | VPC, two public subnets, IGW, route table, ALB and task security groups |
| `modules/service/` | ALB, target group, listener, ECS cluster, task definition, service, log group, IAM roles |
| `validate.py` | Offline parse and cross-reference check |

## Design

The root module only wires variables into `network` and `service`; anything
that creates resources lives in a module. The network module exposes subnet,
security-group and VPC IDs, and the service module consumes them, so the
dependency direction is one-way and visible in `main.tf`.

Traffic path: internet on port 80 → ALB (security group open only on 80) →
target group → Fargate task on `container_port`. The task security group
accepts that port from the ALB security group only. Tasks run in public
subnets with a public IP so they can pull images from ECR without a NAT
gateway; that is the cheapest workable setup for a demo, and the security
groups still block inbound access except from the load balancer.

The task definition writes to a CloudWatch group that takes its name from the
same `${project_name}-${environment}` prefix as everything else, and the
container gets `APP_ENV` and `PORT` from the Terraform variables rather than
hard-coded values.

## Variables

All inputs have defaults, so `terraform plan` works without a tfvars file;
`terraform.tfvars.example` shows how to override them. The ones worth calling
out:

- `container_image` — full image reference; the default is a placeholder
  nginx image so the plan is valid before the Task 5 image is pushed.
- `cpu` / `memory` — validated to Fargate's allowed values (256–4096 units,
  512–30720 MiB); 1024/2048 is 1 vCPU and 2 GiB.
- `desired_count` — 1–10 tasks. Above one task the ALB spreads requests and
  a rolling deployment stays available.
- `project_name` and `environment` — validated to slug and enum
  respectively, since both end up in resource names that AWS restricts.
- `public_subnet_cidrs` and `availability_zones` — must have the same length;
  the validation catches the common mistake of adding a subnet without an AZ.

## Validation

The machine used for this task has no Terraform binary and no AWS account, so
the CLI steps could not be executed here. Two things were done instead.

1. Every file parses with `python-hcl2`, the same HCL grammar Terraform uses.
2. `validate.py` re-implements the checks that matter at this stage:
   - all four required root files exist;
   - every `var.*` in the root is declared;
   - every `module.<name>.<output>` reference matches a declared output;
   - every module call names a real source directory, passes only declared
     inputs, and supplies every input without a default;
   - `terraform.tfvars.example` only sets declared variables.

```
$ cellula_week8/.venv/bin/python Task6_Terraform/validate.py
parsed 10 files, 14 root variables, 20 blocks, 17 resources
OK: syntax parses and cross-references resolve
```

On a machine with Terraform and AWS credentials the normal sequence is:

```
terraform init      # download the aws provider, 5.60.x
terraform validate  # provider schema check
terraform plan -var-file=terraform.tfvars
```

`init` needs the provider registry and `plan` needs credentials for a region
lookup, which is why neither runs offline. `apply` was deliberately not part
of the task and nothing here creates cloud resources by itself.

## Limits

- State is local. For shared use it should move to an S3 backend with locking;
  that needs a bucket and DynamoDB table, so it stays out of this demo.
- No HTTPS listener or certificate; adding one means an ACM certificate and a
  443 listener on the ALB.
- No autoscaling. `desired_count` is fixed; App Auto Scaling on
  `ECSServiceAverageCPUUtilization` would be the next step.
- Tasks in public subnets are fine for a demo, but production should put them
  in private subnets with a NAT gateway or VPC endpoints.
