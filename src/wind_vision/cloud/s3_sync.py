import os
import boto3
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wind_vision.cloud_upload")

def prepare_and_upload_image(img_path: Path, bucket_name: str, s3_client, target_size=(224, 224)):
    """Resize an image and upload it directly to S3 without local caching."""
    try:
        # Load and resize image
        img = cv2.imread(str(img_path))
        if img is None:
            return False
            
        img_resized = cv2.resize(img, target_size)
        
        # Convert to memory buffer (JPG format)
        _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # Create S3 Key (path in bucket)
        s3_key = f"training-data/224x224/{img_path.name}"
        
        # Upload
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=buffer.tobytes(),
            ContentType='image/jpeg'
        )
        return True
    except Exception as e:
        logger.error(f"Error processing {img_path.name}: {e}")
        return False

def sync_dataset_to_s3(local_dir: str, bucket_name: str):
    """Parallel processing and upload of the dataset to S3."""
    s3_client = boto3.client('s3')
    local_path = Path(local_dir)
    images = list(local_path.glob("*.jpg")) + list(local_path.glob("*.png"))
    
    logger.info(f"Starting upload of {len(images)} images to bucket {bucket_name}...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(
            lambda x: prepare_and_upload_image(x, bucket_name, s3_client), 
            images
        ))
    
    success_count = sum(results)
    logger.info(f"Done! {success_count}/{len(images)} images uploaded successfully.")

if __name__ == "__main__":
    # Example usage (after terraform apply)
    # sync_dataset_to_s3("data/raw/webcam", "your-bucket-name")
    pass
