provider "aws" {
  region = "eu-central-1"
}

resource "aws_s3_bucket" "wind_vision_data" {
  bucket = "wind-vision-ml-assets-${random_id.suffix.hex}"
  
  tags = {
    Project = "Wind-Vision"
    Managed = "Terraform"
  }
}

resource "random_id" "suffix" {
  byte_length = 4
}

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "wind-vision-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "S3AccessPolicy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.wind_vision_data.arn,
          "${aws_s3_bucket.wind_vision_data.arn}/*"
        ]
      }
    ]
  })
}

output "s3_bucket_name" {
  value = aws_s3_bucket.wind_vision_data.id
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution_role.arn
}
