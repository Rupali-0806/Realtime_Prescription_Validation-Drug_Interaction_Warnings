# Zaura Health - PostgreSQL RDS Deployment Guide

## 🚀 AWS Free Tier Deployment with RDS PostgreSQL

This guide provides complete instructions for deploying Zaura Health with PostgreSQL RDS database while staying within AWS Free Tier limits.

## 📊 AWS Free Tier Resources Used

### **Compute & Storage**
- **EC2**: 750 hours/month of t2.micro instance
- **RDS**: 750 hours/month of db.t3.micro PostgreSQL
- **EBS**: 30GB General Purpose SSD storage
- **S3**: 5GB storage for backups

### **Database Specifications**
- **Instance**: db.t3.micro (1 vCPU, 1GB RAM)
- **Engine**: PostgreSQL 15.4
- **Storage**: 20GB SSD with encryption
- **Backup**: 7 days retention (free tier maximum)
- **Multi-AZ**: Disabled (to stay in free tier)

### **Expected Monthly Costs**
- **Within Free Tier**: $0-10 (minimal overages)
- **After 12 months**: $15-25/month

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitHub        │    │      EC2        │    │      RDS        │
│   Actions       │───▶│   t2.micro      │◀──▶│  db.t3.micro    │
│   (CI/CD)       │    │   Flask App     │    │  PostgreSQL     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   CloudWatch    │    │ Systems Manager │
                    │   Monitoring    │    │   Credentials   │
                    └─────────────────┘    └─────────────────┘
```

## 📁 Updated File Structure

```
Zaura Health/
├── terraform-ec2/
│   ├── rds.tf                    # RDS PostgreSQL configuration
│   ├── variables.tf              # Updated with DB variables
│   ├── outputs.tf               # RDS connection info
│   └── main.tf                  # EC2 and networking
├── database/
│   └── postgres_manager.py      # PostgreSQL database manager
├── app_rds.py                   # RDS-enabled Flask application
├── Dockerfile.rds               # Container with PostgreSQL support
├── docker-compose.yml           # Updated for RDS
├── requirements.txt             # Added PostgreSQL dependencies
└── scripts/
    ├── setup-ec2.sh             # Updated with PostgreSQL client
    └── deploy.sh                # Updated with RDS connectivity
```

## 🚀 Deployment Steps

### Step 1: Prerequisites

1. **AWS Account Setup**
   - AWS Free Tier account
   - AWS CLI configured with appropriate permissions
   - EC2 Key Pair created

2. **GitHub Repository Setup**
   - Repository with all application files
   - GitHub Actions enabled

3. **Required IAM Permissions**
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": [
           "ec2:*",
           "rds:*",
           "ssm:GetParameter",
           "ssm:PutParameter",
           "iam:PassRole",
           "iam:CreateRole",
           "iam:AttachRolePolicy"
         ],
         "Resource": "*"
       }
     ]
   }
   ```

### Step 2: Configure GitHub Secrets

Add these secrets in **Settings → Secrets → Actions**:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
EC2_SSH_PRIVATE_KEY=your_private_key_content
EC2_SSH_PUBLIC_KEY=your_public_key_content
EC2_KEY_NAME=your_key_pair_name
SECRET_KEY=your_flask_secret_key_for_production
```

### Step 3: Configure Terraform Variables

Create `terraform-ec2/terraform.tfvars`:

```hcl
# Basic Configuration
aws_region = "us-east-1"
project_name = "zaura-health"
environment = "dev"

# EC2 Configuration
instance_type = "t2.micro"
key_name = "your-key-pair-name"
volume_size = 20

# Database Configuration
db_name = "zaura_health"
db_username = "zaura_admin"

# Security
allowed_cidr_blocks = ["0.0.0.0/0"]
ssh_cidr_blocks = ["YOUR_IP/32"]  # Restrict to your IP
db_allowed_cidr_blocks = ["10.0.0.0/16"]
```

### Step 4: Deploy Infrastructure

1. **Manual Terraform Deployment** (First time):
   ```bash
   cd terraform-ec2
   terraform init
   terraform plan -var-file="terraform.tfvars"
   terraform apply -var-file="terraform.tfvars"
   ```

2. **Automated GitHub Actions** (Subsequent deployments):
   ```bash
   git add .
   git commit -m "Deploy with RDS"
   git push origin main
   ```

### Step 5: Monitor Deployment

1. **GitHub Actions Workflow** monitors:
   - Infrastructure provisioning
   - Database deployment
   - Application deployment
   - Health checks

2. **CloudWatch Monitoring** tracks:
   - EC2 system metrics
   - RDS database metrics
   - Application logs
   - Custom application metrics

## 🔧 Database Management

### Connection Information

The application automatically retrieves database credentials from AWS Systems Manager Parameter Store:

- **Parameter**: `/{project_name}/{environment}/database/url`
- **Format**: `postgresql://username:password@endpoint:port/database`

### Database Schema

The application creates these tables automatically:

```sql
-- Users table for authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Drug contributions from community
CREATE TABLE drug_contributions (
    id SERIAL PRIMARY KEY,
    contributor_id INTEGER REFERENCES users(id),
    drug_combination TEXT NOT NULL,
    safety_label VARCHAR(20) NOT NULL,
    dosage_info TEXT,
    notes TEXT,
    confidence_level INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction logging for analytics
CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    input_drugs TEXT[] NOT NULL,
    dosage_info DECIMAL(10,2),
    prediction_result JSONB,
    response_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Manual Database Access

```bash
# Connect via EC2 instance
ssh -i ~/.ssh/your-key.pem ec2-user@your-ec2-ip

# Get database credentials
aws ssm get-parameter --name "/zaura-health/dev/database/url" --with-decryption

# Connect to PostgreSQL
psql "postgresql://username:password@rds-endpoint:5432/zaura_health"
```

## 🔒 Security Configuration

### Network Security
- **VPC**: Private subnets for RDS
- **Security Groups**: 
  - EC2: HTTP(80), HTTPS(443), SSH(22)
  - RDS: PostgreSQL(5432) only from EC2
- **Encryption**: RDS encryption at rest enabled

### Access Control
- **IAM Roles**: EC2 instance role for Parameter Store access
- **Credentials**: Stored in Systems Manager Parameter Store
- **SSH**: Key-based authentication only

### Data Security
- **Passwords**: Hashed using SHA-256
- **Database**: Encrypted storage
- **Backups**: Automated with 7-day retention

## 📊 Monitoring and Maintenance

### Health Checks

**Application Health**:
```bash
curl http://your-ec2-ip/api/health
```

**Database Health**:
```bash
curl http://your-ec2-ip/api/stats
```

### CloudWatch Metrics

**EC2 Metrics**:
- CPU utilization
- Memory usage
- Disk I/O
- Network traffic

**RDS Metrics**:
- Database connections
- CPU utilization
- Storage usage
- Read/Write IOPS

### Log Files

**Application logs**: `/app/logs/app.log`
**Database logs**: CloudWatch Logs group `/zaura-health/postgres`
**Nginx logs**: `/var/log/nginx/`

## 🔄 Backup and Recovery

### Automated Backups
- **RDS**: Daily automated backups (7 days retention)
- **Application**: EC2 AMI snapshots
- **Code**: Git repository

### Manual Backup
```bash
# Database backup
pg_dump "postgresql://user:pass@endpoint:port/db" > backup.sql

# Restore database
psql "postgresql://user:pass@endpoint:port/db" < backup.sql
```

### Disaster Recovery
1. **Infrastructure**: Recreate using Terraform
2. **Database**: Restore from automated backup
3. **Application**: Redeploy from Git repository

## 🚨 Troubleshooting

### Common Issues

**1. Database Connection Failed**
```bash
# Check RDS status
aws rds describe-db-instances --db-instance-identifier zaura-health-postgres-db

# Test network connectivity
telnet rds-endpoint 5432

# Check security groups
aws ec2 describe-security-groups
```

**2. Application Won't Start**
```bash
# Check container logs
docker logs zaura-health-app

# Check database migration
docker logs zaura-health-migration

# Restart services
docker-compose restart
```

**3. Memory Issues (t2.micro)**
```bash
# Check memory usage
free -h

# Restart application to clear memory
docker-compose restart zaura-health
```

**4. RDS Connection Timeout**
```bash
# Check RDS security group
aws ec2 describe-security-groups --group-ids sg-xxxxxxxx

# Verify Parameter Store credentials
aws ssm get-parameter --name "/zaura-health/dev/database/url"
```

### Performance Optimization

**Database Query Optimization**:
- Indexes on frequently queried columns
- Connection pooling (5 connections max)
- Query timeout settings

**Memory Management**:
- Single worker process for Gunicorn
- Garbage collection in Python
- Container memory limits

**Caching Strategy**:
- Nginx static file caching
- Application-level caching for drug suggestions
- Database query result caching

## 🔧 Development Workflow

### Local Development with RDS

1. **Set up local environment**:
   ```bash
   export DATABASE_URL="postgresql://user:pass@rds-endpoint:port/db"
   python app_rds.py
   ```

2. **Database migrations**:
   ```python
   from database.postgres_manager import get_db_manager
   db = get_db_manager()
   # Database schema is created automatically
   ```

### Testing Database Changes

```bash
# Run tests with PostgreSQL
python -m pytest tests/ -v

# Test database connectivity
python -c "from database.postgres_manager import get_db_manager; db = get_db_manager(); print('✓ Connected')"
```

## 📈 Scaling Considerations

### Within Free Tier
- **Vertical scaling**: Optimize code and queries
- **Caching**: Implement Redis (ElastiCache free tier)
- **CDN**: CloudFront for static assets

### Beyond Free Tier
- **Multi-AZ RDS**: High availability
- **Auto Scaling**: EC2 Auto Scaling Groups  
- **Load Balancer**: Application Load Balancer
- **Larger instances**: t3.small, t3.medium

## 💡 Cost Optimization Tips

1. **Stop EC2 when not in use** (development)
2. **Monitor free tier usage** in AWS Console
3. **Set up billing alerts** for $1, $5, $10
4. **Use CloudWatch efficiently** (basic metrics only)
5. **Optimize RDS storage** (start with 20GB)

## 🎯 Production Readiness Checklist

- [ ] SSL certificate configured
- [ ] Domain name pointed to EC2
- [ ] Monitoring alerts set up
- [ ] Backup procedures tested
- [ ] Security groups restricted
- [ ] Database password rotated
- [ ] Application secrets secured
- [ ] Performance testing completed

---

**🚨 Important Notes:**
- Monitor AWS Free Tier usage regularly
- RDS backup window is optimized for minimal impact
- Database credentials are automatically rotated every 90 days
- All components are designed for cost efficiency within free tier limits

**📞 Support Resources:**
- AWS Free Tier FAQ
- PostgreSQL Documentation  
- Flask-SQLAlchemy Documentation
- Terraform AWS Provider Documentation