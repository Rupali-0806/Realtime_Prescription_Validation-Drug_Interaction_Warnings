# Zaura Health - AWS Free Tier Deployment Guide

## Overview
This guide provides complete instructions for deploying the Zaura Health drug interaction prediction application on AWS Free Tier using EC2, optimized for cost-effectiveness and performance within free tier limits.

## Prerequisites

### AWS Account Setup
- AWS Free Tier account (first 12 months)
- AWS CLI configured with access keys
- EC2 Key Pair created in your target region

### Local Development Environment
- Git installed
- GitHub account with repository access
- SSH client (PuTTY for Windows, built-in for Mac/Linux)

### Free Tier Resources Used
- **EC2**: 750 hours/month of t2.micro instances
- **S3**: 5GB storage, 20,000 GET requests, 2,000 PUT requests
- **CloudWatch**: 10 custom metrics, 10 alarms, 5GB log ingestion
- **Data Transfer**: 15GB outbound data transfer

## Repository Structure

```
Zaura Health/
├── terraform-ec2/           # Infrastructure as Code
├── scripts/                 # Deployment automation
├── .github/workflows/       # CI/CD pipelines
├── monitoring/              # CloudWatch configuration
├── models/                  # ML model files
├── static/                  # Web assets
├── templates/               # HTML templates
├── app_ec2.py              # EC2-optimized Flask app
├── Dockerfile.ec2          # Container configuration
├── docker-compose.yml      # Service orchestration
└── requirements.txt        # Python dependencies
```

## Deployment Steps

### Step 1: Infrastructure Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd "Zaura Health"
   ```

2. **Configure Terraform Variables**
   ```bash
   cd terraform-ec2
   cp terraform.tfvars.example terraform.tfvars
   ```
   
   Edit `terraform.tfvars`:
   ```hcl
   region = "us-east-1"  # Choose your preferred region
   key_name = "your-ec2-key-pair-name"
   allowed_cidr = "0.0.0.0/0"  # Restrict to your IP for security
   ```

3. **Deploy Infrastructure**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```
   
   **Expected Output:**
   - EC2 instance (t2.micro)
   - Security groups
   - IAM role with CloudWatch permissions
   - Elastic IP address

### Step 2: GitHub Actions Setup

1. **Add Repository Secrets** (Settings → Secrets → Actions):
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   EC2_SSH_KEY=your_private_key_content
   AWS_REGION=us-east-1
   ```

2. **Trigger Deployment**
   - Push changes to `main` branch
   - GitHub Actions will automatically:
     - Validate Terraform configuration
     - Run application tests
     - Deploy to EC2 instance
     - Configure monitoring

### Step 3: Manual Server Configuration (if needed)

1. **Connect to EC2 Instance**
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   ```

2. **Run Setup Script**
   ```bash
   sudo bash /home/ec2-user/setup-ec2.sh
   ```

3. **Deploy Application**
   ```bash
   sudo bash /home/ec2-user/deploy.sh
   ```

## Application Configuration

### Environment Variables
Set these in your EC2 instance or GitHub secrets:
```bash
export SECRET_KEY="your-production-secret-key"
export AWS_DEFAULT_REGION="us-east-1"
export ENVIRONMENT="production"
```

### Model Files Setup
Ensure these files are present in the `models/` directory:
- `best_enhanced_drug_interaction_model.pth`
- `enhanced_drug_interaction_preprocessor.pkl`
- `enhanced_model_info.pkl`

## Monitoring and Maintenance

### CloudWatch Dashboard
Access CloudWatch in AWS Console to view:
- System metrics (CPU, Memory, Disk, Network)
- Application logs
- Custom application metrics
- Health status alarms

### Log Files
Monitor these log locations on EC2:
- Application logs: `/app/logs/app.log`
- Nginx logs: `/var/log/nginx/`
- Container logs: `/var/log/docker/`

### Health Checks
- Application health: `http://your-ip/api/health`
- System status: `sudo systemctl status docker nginx`

## Cost Optimization

### Free Tier Limits Monitoring
- Set up billing alarms for $1, $5, $10
- Monitor usage in AWS Cost Explorer
- Use AWS Free Tier usage alerts

### Resource Optimization
- Single t2.micro instance only
- Minimal CloudWatch retention (3-7 days)
- S3 lifecycle policies for log archival
- Stop instance during development breaks

### Expected Monthly Costs
- **Within Free Tier**: $0-5 (minimal data transfer overages)
- **After Free Tier**: $8-15/month for t2.micro + storage

## Security Best Practices

### Network Security
- Security group allows only necessary ports (80, 443, 22)
- SSH access restricted to your IP address
- HTTPS enabled with SSL certificates

### Application Security
- Environment variables for sensitive data
- Regular security updates via automation
- Model files stored securely in S3

### Access Control
- IAM roles with minimal permissions
- No hardcoded credentials in code
- SSH key-based authentication only

## Troubleshooting

### Common Issues

1. **Application Won't Start**
   ```bash
   sudo docker logs zaura-health-app
   sudo systemctl status nginx
   ```

2. **Model Loading Errors**
   ```bash
   # Check model files exist
   ls -la /app/models/
   # Check application logs
   tail -f /app/logs/app.log
   ```

3. **Memory Issues on t2.micro**
   ```bash
   # Check memory usage
   free -h
   # Restart services if needed
   sudo systemctl restart docker
   ```

4. **CloudWatch Not Receiving Data**
   ```bash
   sudo systemctl status amazon-cloudwatch-agent
   sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -query-config
   ```

### Performance Optimization

1. **Memory Management**
   - Application uses garbage collection
   - Single worker process to minimize memory
   - Efficient PyTorch model loading

2. **Response Time**
   - Nginx caching enabled
   - Gzip compression for static files
   - Efficient drug combination processing

## Scaling Considerations

### Vertical Scaling (Within Free Tier)
- Optimize application code for memory efficiency
- Use efficient data structures
- Implement caching strategies

### Horizontal Scaling (Paid Tiers)
- Application Load Balancer
- Auto Scaling Groups
- RDS for database scaling
- ElastiCache for Redis caching

## API Documentation

### Health Check
```bash
curl http://your-instance-ip/api/health
```

### Drug Interaction Prediction
```bash
curl -X POST http://your-instance-ip/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "drugs": ["Aspirin", "Warfarin", "Ibuprofen"],
    "dosage": 2.5
  }'
'```

### Drug Suggestions
```bash
curl "http://your-instance-ip/api/drug-suggestions?q=asp"
```

## Backup and Recovery

### Automated Backups
- EBS snapshots for instance storage
- S3 versioning for model files
- Database exports to S3

### Disaster Recovery
1. Terraform configuration in version control
2. Application code in GitHub
3. Model files backed up to S3
4. Infrastructure can be recreated in minutes

## Development Workflow

### Local Testing
```bash
python -m pytest tests/
docker build -f Dockerfile.ec2 -t zaura-health .
docker run -p 5000:5000 zaura-health
```

### Staging Deployment
- Use separate AWS account or region
- Test with production-like data
- Validate monitoring and alerting

### Production Deployment
- Deploy via GitHub Actions
- Monitor application health
- Verify all services running

## Support and Maintenance

### Regular Tasks
- Weekly: Check CloudWatch metrics and logs
- Monthly: Review AWS costs and usage
- Quarterly: Update dependencies and security patches

### Updates
- Model updates: Replace files in S3, restart application
- Code updates: Push to GitHub, automatic deployment
- Infrastructure updates: Modify Terraform, apply changes

### Contact and Resources
- AWS Free Tier Documentation
- CloudWatch pricing calculator  
- EC2 instance types comparison
- GitHub Actions marketplace

---

**Note**: This deployment is optimized for AWS Free Tier usage. Monitor your usage regularly to avoid unexpected charges. Consider upgrading to paid tiers for production workloads requiring higher availability or performance.