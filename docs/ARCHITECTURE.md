# System Architecture - Wind-Vision

## Overview
Wind-Vision is an AI-driven system for real-time wind speed estimation from environmental webcam streams. It transforms unstructured visual data into actionable meteorological insights.

## System Components

### 1. Data Ingestion (The Fetcher)
- **Technology:** Playwright (Headless Browser)
- **Role:** Navigates to weather portals, bypasses cookie consent, and captures raw imagery.
- **Architectural Policy:** Must be stateless and handle network timeouts gracefully.

### 2. Data Processing (OCR Pipeline)
- **Technology:** EasyOCR
- **Role:** Extracts ground-truth wind speed from image overlays to build a labeled dataset.
- **Data Contract:** Output is a standardized CSV (`processed_wind.csv`) with validated schemas.

### 3. ML Pipeline (Training & Inference)
- **Architecture:** ResNet-18 (Transfer Learning)
- **Preprocessing:** localized water-cropping and alpha-masking to prevent "shortcut learning" from OCR overlays.
- **Tracking:** MLflow experiment registry for reproducibility.

### 4. Serving Layer (FastAPI)
- **Role:** Encapsulates the model behind a RESTful API.
- **Capability:** Single-image inference for downstream consumers.

## Information Flow
`Webcam -> fetcher.py -> raw_images -> extract_wind.py -> CSV -> train.py -> model.pth -> api/server.py -> JSON Result`

## Key Architecture Decisions (ADRs)
- **ADR 1: Local Masking:** Decided to mask the OCR area during training to ensure the model learns wave textures, not text.
- **ADR 2: MLflow Integration:** Used MLflow instead of manual logging to comply with industry standards for model auditing.
- **ADR 3: FastAPI:** Selected over Flask for modern async support and auto-generating OpenAPI documentation.
