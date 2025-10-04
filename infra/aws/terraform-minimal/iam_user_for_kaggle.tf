# IAM User for Kaggle/External access to MLFlow S3 artifacts
# This creates an IAM user with access keys that can be used from Kaggle notebooks

# IAM User for external access (Kaggle)
resource "aws_iam_user" "kaggle_mlflow" {
  name = "${local.project_name}-kaggle-user"
  path = "/mlflow/"

  tags = merge(local.common_tags, {
    Name        = "${local.project_name}-kaggle-user"
    Description = "IAM user for Kaggle notebooks to access MLFlow S3 artifacts"
  })
}

# IAM Policy for S3 access (same permissions as EC2 role)
resource "aws_iam_user_policy" "kaggle_s3_access" {
  name = "${local.project_name}-kaggle-s3-policy"
  user = aws_iam_user.kaggle_mlflow.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow.arn,
          "${aws_s3_bucket.mlflow.arn}/*"
        ]
      }
    ]
  })
}

# Create Access Key for the user
resource "aws_iam_access_key" "kaggle_mlflow" {
  user = aws_iam_user.kaggle_mlflow.name
}

# Output the Access Key ID (Secret Key will be in terraform state)
output "kaggle_aws_access_key_id" {
  description = "AWS Access Key ID for Kaggle notebooks"
  value       = aws_iam_access_key.kaggle_mlflow.id
}

output "kaggle_aws_secret_access_key" {
  description = "AWS Secret Access Key for Kaggle notebooks (sensitive!)"
  value       = aws_iam_access_key.kaggle_mlflow.secret
  sensitive   = true
}

output "kaggle_s3_bucket_name" {
  description = "S3 bucket name for MLFlow artifacts"
  value       = aws_s3_bucket.mlflow.bucket
}

# Instructions output
output "kaggle_setup_instructions" {
  description = "Instructions for setting up Kaggle Secrets"
  value = <<-EOT
  
  ╔════════════════════════════════════════════════════════════════╗
  ║           Kaggle Secrets Setup Instructions                    ║
  ╚════════════════════════════════════════════════════════════════╝
  
  1. Get your AWS credentials:
     terraform output kaggle_aws_access_key_id
     terraform output -raw kaggle_aws_secret_access_key
  
  2. Add them to Kaggle Secrets:
     - Go to your Kaggle Notebook
     - Click "Add-ons" → "Secrets"
     - Add secret: AWS_ACCESS_KEY_ID = <value from step 1>
     - Add secret: AWS_SECRET_ACCESS_KEY = <value from step 1>
  
  3. S3 Bucket: ${aws_s3_bucket.mlflow.bucket}
  4. AWS Region: ${var.aws_region}
  
  ⚠️  Keep these credentials secure!
  ⚠️  Never commit them to git or share publicly!
  
  EOT
}






