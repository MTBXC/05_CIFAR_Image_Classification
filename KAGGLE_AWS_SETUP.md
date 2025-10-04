# 🔐 AWS Credentials Setup for Kaggle Notebooks

This guide explains how to create and configure AWS credentials so your Kaggle notebooks can save MLFlow artifacts to your S3 bucket.

## 🤔 Why do I need this?

Your MLFlow server on EC2 uses **IAM Role** to access S3 (no keys needed). However, Kaggle notebooks run **outside AWS** and need **Access Keys** to authenticate.

## 📋 Current Setup

- **S3 Bucket**: `mlflow-minimal-z01phmk8`
- **AWS Region**: `eu-north-1`
- **MLFlow Server**: `http://13.51.104.28:5000`

## 🚀 Step-by-Step Instructions

### Step 1: Create IAM User with Access Keys

Navigate to the Terraform directory and apply the new configuration:

```bash
cd infra/aws/terraform-minimal

# Apply the new IAM user configuration
terraform apply
```

This will create:
- IAM User: `mlflow-minimal-kaggle-user`
- Access Key ID and Secret Access Key
- S3 permissions (same as EC2 role)

### Step 2: Get Your Credentials

After `terraform apply`, get your credentials:

```bash
# Get Access Key ID
terraform output kaggle_aws_access_key_id

# Get Secret Access Key (sensitive!)
terraform output -raw kaggle_aws_secret_access_key
```

**⚠️ IMPORTANT:** Copy these values immediately! The Secret Access Key is only shown once.

### Step 3: Add Credentials to Kaggle Secrets

1. **Open your Kaggle Notebook**
   - Go to: https://www.kaggle.com/code

2. **Access Secrets Menu**
   - Click **"Add-ons"** (right sidebar)
   - Select **"Secrets"**

3. **Add AWS_ACCESS_KEY_ID**
   - Click **"Add a new secret"**
   - Label: `AWS_ACCESS_KEY_ID`
   - Value: Paste the Access Key ID from Step 2
   - Click **"Add"**

4. **Add AWS_SECRET_ACCESS_KEY**
   - Click **"Add a new secret"** again
   - Label: `AWS_SECRET_ACCESS_KEY`
   - Value: Paste the Secret Access Key from Step 2
   - Click **"Add"**

5. **Enable Secrets in Your Notebook**
   - Toggle **ON** for both secrets
   - You should see: 🔓 AWS_ACCESS_KEY_ID (Enabled)
   - You should see: 🔓 AWS_SECRET_ACCESS_KEY (Enabled)

### Step 4: Use Credentials in Kaggle Notebook

Add this **Cell 0** to your Kaggle notebook (before Cell 1):

```python
# ====================================================================
# 🔐 Cell 0: AWS S3 Configuration for MLFlow Artifacts
# ====================================================================

import os

# Load AWS credentials from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    
    os.environ['AWS_ACCESS_KEY_ID'] = user_secrets.get_secret("AWS_ACCESS_KEY_ID")
    os.environ['AWS_SECRET_ACCESS_KEY'] = user_secrets.get_secret("AWS_SECRET_ACCESS_KEY")
    os.environ['AWS_DEFAULT_REGION'] = 'eu-north-1'  # Your MLFlow S3 region
    
    print("✅ AWS credentials loaded from Kaggle Secrets")
    print(f"✅ Region: {os.environ['AWS_DEFAULT_REGION']}")
    print("✅ MLFlow can now save artifacts to S3!")
    
except Exception as e:
    print(f"❌ Failed to load AWS credentials: {e}")
    print("⚠️  MLFlow will work, but artifacts may not be saved to S3")
```

## 🧪 Test the Configuration

Test if credentials work:

```python
import boto3

try:
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket='mlflow-minimal-z01phmk8', MaxKeys=1)
    print("✅ S3 connection successful!")
except Exception as e:
    print(f"❌ S3 connection failed: {e}")
```

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Your Setup                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐         ┌──────────────┐                 │
│  │   Kaggle    │         │   EC2 Server │                 │
│  │  Notebook   │────────▶│   (MLFlow)   │                 │
│  └─────────────┘  HTTP   └──────────────┘                 │
│        │                        │                          │
│        │ AWS Access Keys        │ IAM Role                │
│        │                        │                          │
│        ▼                        ▼                          │
│  ┌─────────────────────────────────────┐                  │
│  │          S3 Bucket                  │                  │
│  │   mlflow-minimal-z01phmk8           │                  │
│  │   (MLFlow Artifacts Storage)        │                  │
│  └─────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Flow:
1. Kaggle sends logs/metrics to MLFlow Server via HTTP
2. MLFlow Server stores artifacts in S3 using IAM Role
3. Kaggle can also directly upload to S3 using Access Keys
```

## 🔒 Security Best Practices

1. **Never commit credentials to git**
   - The `.gitignore` already excludes terraform state files
   - Never hardcode keys in notebooks

2. **Use Kaggle Secrets** (as shown above)
   - Don't print credentials in notebook output
   - Don't share notebooks with secrets enabled

3. **Rotate keys periodically**
   ```bash
   # Destroy old key and create new one
   terraform apply -replace="aws_iam_access_key.kaggle_mlflow"
   ```

4. **Limit permissions**
   - The IAM user only has access to the MLFlow S3 bucket
   - No other AWS resources can be accessed

## 🗑️ Cleanup

If you no longer need Kaggle access:

```bash
cd infra/aws/terraform-minimal

# Remove the IAM user and keys
rm iam_user_for_kaggle.tf
terraform apply
```

## ❓ Troubleshooting

### Problem: "No module named 'kaggle_secrets'"
**Solution:** You're running locally, not in Kaggle. Use direct environment variables or skip AWS config.

### Problem: "Access Denied" when accessing S3
**Solution:** 
1. Check if secrets are enabled in Kaggle
2. Verify credentials with: `terraform output kaggle_aws_access_key_id`
3. Ensure region is correct: `eu-north-1`

### Problem: "Bucket does not exist"
**Solution:** Get bucket name: `terraform output kaggle_s3_bucket_name`

### Problem: MLFlow works but artifacts not in S3
**Solution:** This is normal if AWS credentials are not configured. Artifacts will be stored locally on MLFlow server.

## 📚 Additional Resources

- [Kaggle Secrets Documentation](https://www.kaggle.com/docs/notebooks#secrets)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [MLFlow S3 Artifact Store](https://mlflow.org/docs/latest/tracking.html#amazon-s3-and-s3-compatible-storage)

## ✅ Summary

After following these steps:
- ✅ IAM User created with S3 access
- ✅ Access Keys stored in Kaggle Secrets
- ✅ Kaggle notebooks can log to MLFlow
- ✅ Artifacts automatically saved to S3
- ✅ Secure and production-ready setup






