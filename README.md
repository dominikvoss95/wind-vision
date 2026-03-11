# Wind-Vision 🌬️📸

**Real-time Wind Speed Estimation from Environmental Webcam Streams.**  
An end-to-end MLOps platform that transforms unstructured visual data into actionable meteorological insights.

---

## 🚀 MLOps & Platform Features

This project is built with a focus on **Production ML Best Practices**, addressing challenges in scalability, cost-efficiency, and regulatory compliance.

### 🏗️ Infrastructure as Code (IaC) & Scalability
- **Terraform:** Infrastructure defined as code to provision AWS S3 buckets and IAM Roles for SageMaker training.
- **Kubernetes (k3d):** The inference API (FastAPI) is containerized and deployable to a K8s cluster with resource limits, load balancing, and replicas for high availability.
- **Docker:** Multi-stage builds using non-root users to ensure container security and "Least Privilege" execution.

### 💰 Cost Optimization 
- **Managed Spot Training:** Configured for AWS SageMaker to reduce compute costs by up to 90%.
- **Intelligent Data Sync:** Custom S3 synchronization script that resizes images in-memory (224x224) before upload, reducing storage and data transfer costs by >95%.
- **Checkpointing:** Native support for training resumes to prevent data loss during Spot Instance interruptions.

### 🧪 ML Pipeline & Security
- **Data Ingestion:** Playwright-based headless browser automation for robust historical data collection.
- **Automated Labeling:** Integrated OCR pipeline (EasyOCR) to extract ground-truth from image overlays.
- **Model Security:** Secure model serialization using `weights_only=True` to prevent arbitrary code execution artifacts.
- **CI/CD:** Automated linting and testing via GitHub Actions on every push.

---

## 🛠️ Architecture

```mermaid
graph LR
    A[Webcam] --> B[Playwright Fetcher]
    B --> C[Raw Images]
    C --> D[OCR Extraction]
    D --> E[Processed Dataset]
    E --> F[SageMaker Spot Training]
    F --> G[Model Registry]
    G --> H[FastAPI on Kubernetes]
    H --> I[JSON Results]
```

## 📦 Project Structure

```bash
src/wind_vision/
├── api/             # FastAPI Serving layer
├── cloud/           # AWS S3 Sync & Data Engineering utilities
├── core/            # Configuration & Logging
├── data/            # Scrapers & OCR labeling
└── models/          # PyTorch architectures & training loops
terraform/           # IaC for AWS Resources
k8s/                 # Kubernetes Manifests
```

## ⚡ Quick Start

```bash
# 1. Setup Environment
make setup

# 2. Infrastructure (Optional)
cd terraform && terraform init && terraform apply

# 3. Local Inference in Kubernetes
docker build -t wind-vision-api:v1 .
k3d image import wind-vision-api:v1 -c wind-vision-cluster
kubectl apply -f k8s/deployment.yaml

# 4. Access API
curl http://localhost:8080/
```

## 📚 Tech Stack

- **ML & Computer Vision:** PyTorch, Torchvision, OpenCV, EasyOCR
- **Ops:** Terraform, Kubernetes, Docker, GitHub Actions
- **Backend:** FastAPI, Uvicorn, Playwright
- **Storage:** AWS S3

