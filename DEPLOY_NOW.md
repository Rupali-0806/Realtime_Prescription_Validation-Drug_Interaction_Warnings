# 🚀 IMMEDIATE DEPLOYMENT CHECKLIST
**Ready to deploy Zaura Health with PostgreSQL RDS via GitHub Actions!**

## ⚡ STEP 1: GitHub Repository Setup

### Required GitHub Secrets
Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets (COPY THE NAMES EXACTLY):

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
EC2_SSH_PRIVATE_KEY
EC2_SSH_PUBLIC_KEY
EC2_KEY_NAME
SECRET_KEY
```

### Secret Values Needed:

1. **AWS_ACCESS_KEY_ID**
   - Your AWS access key ID
   - Example: `AKIAIOSFODNN7EXAMPLE`

2. **AWS_SECRET_ACCESS_KEY**
   - Your AWS secret access key
   - Example: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

3. **EC2_SSH_PRIVATE_KEY**
   - Your EC2 key pair private key content
   - Include the full content including:
   ```
   -----BEGIN RSA PRIVATE KEY-----
   [your private key content]
   -----END RSA PRIVATE KEY-----
   ```

4. **EC2_SSH_PUBLIC_KEY**
   - Your EC2 key pair public key content
   - Example: `ssh-rsa AAAAB3NzaC1yc2EAAAA... your-key-name`

5. **EC2_KEY_NAME**
   - Name of your EC2 key pair (without .pem extension)
   - Example: `zaura-health-key`

6. **SECRET_KEY**
   - Flask application secret key for production
   - Generate a strong random key, example:
   ```bash
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

## ⚡ STEP 2: AWS Account Prerequisites

### Required AWS Services Access:
- [ ] **EC2** - Full access for instance creation
- [ ] **RDS** - Full access for PostgreSQL database
- [ ] **VPC** - For networking setup
- [ ] **IAM** - For role creation
- [ ] **Systems Manager** - For parameter store
- [ ] **CloudWatch** - For monitoring

### AWS CLI Test (Optional - verify locally):
```bash
aws sts get-caller-identity
aws ec2 describe-regions --region us-east-1
```

## ⚡ STEP 3: Deploy NOW! 

### Option A: Push to GitHub (RECOMMENDED)
```bash
# In your local repository
git add .
git commit -m "🚀 Deploy Zaura Health with PostgreSQL RDS"
git push origin main
```

### Option B: Manual Terraform (Alternative)
```bash
cd terraform-ec2
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"
```

## ⚡ STEP 4: Monitor Deployment

### GitHub Actions Workflow:
1. Go to your GitHub repository
2. Click **Actions** tab
3. Watch the workflow: "Deploy EC2 with RDS"
4. Monitor these jobs:
   - ✅ **Test** (2-3 minutes)
   - ✅ **Infrastructure** (8-12 minutes)
   - ✅ **Deploy** (3-5 minutes)

### Expected Timeline:
- **Total Deployment**: 15-20 minutes
- **RDS Creation**: 8-10 minutes (longest step)
- **EC2 Setup**: 3-5 minutes
- **Application Deploy**: 2-3 minutes

## ⚡ STEP 5: Verify Deployment

### Once Complete, Test These URLs:
```bash
# Replace YOUR_EC2_IP with actual IP from GitHub Actions output

# Health Check
curl http://YOUR_EC2_IP/api/health

# Application Home
curl http://YOUR_EC2_IP/

# Drug Prediction API
curl -X POST http://YOUR_EC2_IP/api/predict \
  -H "Content-Type: application/json" \
  -d '{"drugs": ["Aspirin", "Warfarin"], "dosage": 2.0}'

# Database Statistics
curl http://YOUR_EC2_IP/api/stats
```

## 🚨 TROUBLESHOOTING

### If Deployment Fails:

1. **Check GitHub Actions Logs**
   - Click on failed job
   - Expand error sections
   - Common issues: AWS permissions, key pair, region

2. **Common Fixes**:
   - Verify AWS credentials are correct
   - Ensure EC2 key pair exists in us-east-1 region
   - Check AWS service limits (should be fine for free tier)

3. **Manual Retry**:
   ```bash
   git commit --allow-empty -m "🔄 Retry deployment"
   git push origin main
   ```

## 📊 EXPECTED RESOURCES CREATED

### AWS Infrastructure:
- **EC2**: 1 x t2.micro instance
- **RDS**: 1 x db.t3.micro PostgreSQL database
- **VPC**: Custom VPC with 2 subnets
- **Security Groups**: Web and database security groups
- **IAM**: EC2 instance role for parameter access
- **EIP**: Elastic IP for stable access
- **Parameter Store**: Database credentials

### Estimated Costs:
- **First 12 months**: $0-5/month (Free Tier)
- **After free tier**: $20-25/month
- **RDS**: ~$12/month after free tier
- **EC2**: ~$8/month after free tier

## 🎯 SUCCESS INDICATORS

✅ GitHub Actions workflow completes successfully
✅ EC2 instance running and accessible
✅ RDS database created and connected
✅ Application responds to health checks
✅ Database logging working (check /api/stats)
✅ All security groups configured properly

---

**🚀 READY TO DEPLOY?** 
**Push to GitHub main branch and watch the magic happen!**

**⏱️ Deployment ETA: 15-20 minutes**
**💰 Cost: FREE (within AWS Free Tier)**
**🔒 Security: Production-ready with RDS encryption**