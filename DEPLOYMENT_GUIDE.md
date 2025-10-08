# Zaura Health - AWS Deployment Guide

## 🚀 Complete AWS Deployment with ECS, S3, and Terraform

This guide provides step-by-step instructions to deploy your Zaura Health application to AWS using ECS (Elastic Container Service), S3, and Terraform with GitHub Actions CI/CD.

## 📋 Prerequisites

### 1. AWS Account Setup
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Domain name (optional, for HTTPS)

### 2. Local Development Tools
- Docker Desktop installed
- Terraform >= 1.5.0 installed
- Git repository set up

### 3. Required AWS Services
- Amazon ECS (Elastic Container Service)
- Amazon ECR (Elastic Container Registry)
- Amazon S3 (Simple Storage Service)
- Amazon VPC (Virtual Private Cloud)
- Application Load Balancer (ALB)
- CloudWatch for logging and monitoring

## 🔧 Step-by-Step Deployment

### Step 1: AWS IAM Setup

1. **Create IAM User for GitHub Actions**
   ```bash
   # Create IAM user
   aws iam create-user --user-name github-actions-zaura-health
   
   # Attach policy (use the policy from config/secrets.example)
   aws iam attach-user-policy --user-name github-actions-zaura-health --policy-arn arn:aws:iam::YOUR-ACCOUNT:policy/ZauraHealthDeploymentPolicy
   
   # Create access keys
   aws iam create-access-key --user-name github-actions-zaura-health
   ```

2. **Save the Access Key ID and Secret Access Key** for GitHub secrets

### Step 2: Terraform State Backend (Optional but Recommended)

1. **Create S3 bucket for Terraform state**
   ```bash
   aws s3 mb s3://zaura-health-terraform-state-YOUR-UNIQUE-ID
   aws s3api put-bucket-versioning --bucket zaura-health-terraform-state-YOUR-UNIQUE-ID --versioning-configuration Status=Enabled
   ```

2. **Create DynamoDB table for state locking**
   ```bash
   aws dynamodb create-table \
     --table-name zaura-health-terraform-locks \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --provisioned-throughput ReadCapacityUnits=1,WriteCapacityUnits=1
   ```

3. **Update terraform/main.tf** - uncomment and configure the backend block:
   ```hcl
   backend "s3" {
     bucket         = "zaura-health-terraform-state-YOUR-UNIQUE-ID"
     key            = "infrastructure/terraform.tfstate"
     region         = "us-east-1"
     dynamodb_table = "zaura-health-terraform-locks"
     encrypt        = true
   }
   ```

### Step 3: GitHub Repository Setup

1. **Configure GitHub Secrets**
   Go to your GitHub repository → Settings → Secrets and variables → Actions

   Add the following secrets:
   ```
   AWS_ACCESS_KEY_ID=your-access-key-id
   AWS_SECRET_ACCESS_KEY=your-secret-access-key
   ```

2. **Configure Environment Variables**
   - Create environments: `development` and `production`
   - Set protection rules for production environment

### Step 4: Infrastructure Deployment

1. **Clone and Prepare Repository**
   ```bash
   git clone your-repository-url
   cd zaura-health
   ```

2. **Update Terraform Variables**
   Edit `terraform/terraform.tfvars`:
   ```hcl
   # Update these values for your deployment
   aws_region = "us-east-1"  # or your preferred region
   environment = "dev"
   project_name = "zaura-health"
   
   # If you have a domain and SSL certificate
   certificate_arn = "arn:aws:acm:us-east-1:123456789012:certificate/your-cert-id"
   domain_name = "your-domain.com"
   ```

3. **Initial Infrastructure Deployment**
   ```bash
   cd terraform
   terraform init
   terraform plan -var-file="terraform.tfvars"
   terraform apply -var-file="terraform.tfvars"
   ```

4. **Note the Outputs**
   After successful deployment, note:
   - ECR Repository URL
   - ALB DNS Name
   - S3 Bucket Names

### Step 5: Container and Model Setup

1. **Upload Model Files to S3** (if using S3 for models)
   ```bash
   # Get S3 bucket name from Terraform output
   S3_BUCKET=$(terraform output -raw s3_bucket_models)
   
   # Upload model files
   aws s3 sync models/ s3://${S3_BUCKET}/models/
   ```

2. **Build and Push Initial Docker Image**
   ```bash
   # Get ECR repository URL from Terraform output
   ECR_REPO=$(terraform output -raw ecr_repository_url)
   
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${ECR_REPO}
   
   # Build and push image
   docker build -t ${ECR_REPO}:latest .
   docker push ${ECR_REPO}:latest
   ```

### Step 6: CI/CD Pipeline Activation

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add AWS deployment configuration"
   git push origin main
   ```

2. **Monitor GitHub Actions**
   - Go to your repository → Actions tab
   - Watch the deployment workflow
   - Check for any errors in the logs

### Step 7: Verification

1. **Check ECS Service**
   ```bash
   aws ecs describe-services --cluster zaura-health-dev-cluster --services zaura-health-dev-service
   ```

2. **Test Application**
   ```bash
   # Get ALB DNS name
   ALB_DNS=$(aws elbv2 describe-load-balancers --names zaura-health-dev-alb --query 'LoadBalancers[0].DNSName' --output text)
   
   # Test health endpoint
   curl http://${ALB_DNS}/api/health
   ```

3. **Access Application**
   - Open browser to `http://YOUR-ALB-DNS-NAME`
   - Test drug interaction predictions

## 📊 Monitoring and Logging

### CloudWatch Dashboard
- Access AWS Console → CloudWatch → Dashboards
- View "zaura-health-dev-dashboard"
- Monitor CPU, memory, request count, and response times

### Application Logs
- AWS Console → CloudWatch → Log groups
- View `/aws/ecs/zaura-health-dev` log group
- Use CloudWatch Insights for log analysis

### Alerts
- SNS topic created for alerts: `zaura-health-dev-alerts`
- Configure email notifications by subscribing to the SNS topic

## 🔄 Environment Management

### Development Environment
- Deployed from `develop` branch
- Lower resource allocation
- Detailed logging enabled

### Production Environment
- Deployed from `main` branch
- Higher resource allocation
- Optimized for performance
- Requires manual approval for deployments

## 🛠️ Maintenance and Updates

### Application Updates
1. Make code changes
2. Commit and push to appropriate branch
3. GitHub Actions automatically builds and deploys
4. Monitor deployment in Actions tab

### Infrastructure Updates
1. Update Terraform files
2. Commit changes to `terraform/` directory
3. GitHub Actions runs `terraform plan` on PRs
4. Merge to apply changes

### Model Updates
1. Update model files in `models/` directory
2. Commit and push changes
3. GitHub Actions syncs files to S3
4. Restart ECS service to load new models

## 🚨 Troubleshooting

### Common Issues

#### 1. ECS Service Fails to Start
```bash
# Check ECS service events
aws ecs describe-services --cluster CLUSTER-NAME --services SERVICE-NAME

# Check task logs
aws logs get-log-events --log-group-name /aws/ecs/zaura-health-dev --log-stream-name STREAM-NAME
```

#### 2. Model Loading Errors
```bash
# Check if model files exist in S3
aws s3 ls s3://BUCKET-NAME/models/

# Verify container has S3 access
# Check IAM role permissions
```

#### 3. Load Balancer Health Checks Failing
```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn TARGET-GROUP-ARN

# Verify security group rules
aws ec2 describe-security-groups --group-ids SECURITY-GROUP-ID
```

#### 4. GitHub Actions Failures
- Check AWS credentials in GitHub secrets
- Verify IAM permissions
- Review action logs for specific errors

### Performance Optimization

#### 1. Auto Scaling
- ECS service auto-scales based on CPU/memory usage
- Adjust thresholds in `terraform/ecs.tf`

#### 2. Cost Optimization
- Use Fargate Spot for development environments
- Implement S3 lifecycle policies for old model versions
- Monitor CloudWatch costs

#### 3. Security Best Practices
- Regularly rotate AWS access keys
- Use AWS Secrets Manager for sensitive configuration
- Enable VPC Flow Logs for network monitoring
- Regular security scanning with tools like AWS Inspector

## 📈 Scaling Considerations

### Horizontal Scaling
- ECS auto-scaling handles traffic increases
- Application Load Balancer distributes traffic
- Consider using CloudFront CDN for static assets

### Vertical Scaling
- Adjust container CPU/memory in `terraform/variables.tf`
- Use larger instance types if needed
- Monitor resource utilization in CloudWatch

### Database Integration (Future)
- Add RDS PostgreSQL for persistent storage
- Implement database migrations
- Use read replicas for high availability

## 💰 Cost Estimation

### Monthly Cost Breakdown (Development)
- ECS Fargate: ~$30-50/month
- Application Load Balancer: ~$20/month
- S3 Storage: ~$5-10/month
- CloudWatch Logs: ~$5-10/month
- NAT Gateway: ~$45/month
- **Total: ~$105-135/month**

### Production Scaling
- Multiple AZs: +50% infrastructure cost
- Higher capacity: +100-200% compute cost
- Monitoring and alerting: +$20-30/month

## 📞 Support and Resources

### AWS Documentation
- [ECS Developer Guide](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/)
- [GitHub Actions AWS Guide](https://github.com/aws-actions)

### Monitoring Resources
- CloudWatch Dashboard templates
- Custom metrics for ML model performance
- Integration with external monitoring tools (DataDog, New Relic)

---

## 🎯 Next Steps

1. **SSL/TLS Setup**: Obtain ACM certificate for HTTPS
2. **Custom Domain**: Configure Route 53 for custom domain
3. **Database Integration**: Add RDS for persistent storage
4. **CDN Setup**: Configure CloudFront for better performance
5. **Backup Strategy**: Implement automated backups
6. **Disaster Recovery**: Set up multi-region deployment

**Congratulations!** 🎉 Your Zaura Health application is now deployed on AWS with professional DevOps practices, monitoring, and CI/CD pipeline.