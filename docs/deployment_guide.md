# 🚀 NFL Analytics Engine: Production Deployment Guide

**Deploy your trained model to production with FastAPI, Docker, and Cloud platforms**

---

## 📋 Overview

This guide covers deploying the trained NFL model for production inference:

- **API Development**: FastAPI REST API
- **Containerization**: Docker packaging
- **Cloud Deployment**: Google Cloud Run, AWS Lambda
- **Monitoring**: Prometheus, Grafana
- **Scaling**: Auto-scaling and load balancing
- **CI/CD**: Automated deployment pipelines

Production runs export TorchScript/ONNX artifacts to `outputs/exported_models/` and record the exact config in `outputs/nfl_production_v2_config.json`, ensuring deploys are tied to reproducible training runs.

---

## 🏗️ Architecture

```
Client Request
    ↓
Load Balancer
    ↓
API Server (FastAPI)
    ↓
Model Inference (TorchScript/ONNX)
    ↓
Response (JSON)
```

---

## 📦 Step 1: Create Inference API

### `inference_api.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="NFL Analytics API",
    description="Production API for NFL player trajectory prediction",
    version="1.0.0"
)

# Load model (TorchScript for production)
MODEL_PATH = "outputs/exported_models/nfl_production_v2_torchscript.pt"
model = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    try:
        model = torch.jit.load(MODEL_PATH)
        model.eval()
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Request/Response models
class PlayerState(BaseModel):
    x: float
    y: float
    s: float  # speed
    a: float  # acceleration
    dis: float  # distance
    o: float  # orientation
    dir: float  # direction

class PredictionRequest(BaseModel):
    players: List[PlayerState]
    game_context: Dict[str, float]  # down, distance, etc.

class TrajectoryPrediction(BaseModel):
    player_id: int
    predicted_positions: List[List[float]]  # [[x1,y1], [x2,y2], ...]
    confidence: float

class PredictionResponse(BaseModel):
    predictions: List[TrajectoryPrediction]
    inference_time_ms: float

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "NFL Analytics Engine v1.0",
        "endpoints": ["/predict", "/health", "/metrics"]
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict player trajectories
    
    Args:
        request: Player states and game context
        
    Returns:
        Trajectory predictions for each player
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Preprocess input
        # (Convert request to graph data format)
        # This is simplified - implement full preprocessing
        
        # Run inference
        with torch.no_grad():
            # predictions = model(graph_data)
            # For demo, return dummy predictions
            predictions = []
            for i, player in enumerate(request.players):
                pred = TrajectoryPrediction(
                    player_id=i,
                    predicted_positions=[[player.x + j, player.y + j*0.5] for j in range(10)],
                    confidence=0.85
                )
                predictions.append(pred)
        
        inference_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions,
            inference_time_ms=inference_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    return {
        "model_version": "1.0.0",
        "total_predictions": 0,  # Track in production
        "avg_inference_time_ms": 0.0,
        "error_rate": 0.0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Test Locally

```bash
# Run API
python inference_api.py

# Test in another terminal
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "players": [
      {"x": 50, "y": 26, "s": 5.0, "a": 0.5, "dis": 1.2, "o": 90, "dir": 90}
    ],
    "game_context": {"down": 1, "distance": 10}
  }'
```

---

## 🐳 Step 2: Containerize with Docker

### `Dockerfile`

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install project
RUN pip install -e .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
.venv
venv/
.git
.gitignore
*.md
tests/
notebooks/
.pytest_cache
lightning_logs/
checkpoints/
mlruns/
*.log
```

### Build and Test

```bash
# Build image
docker build -t nfl-analytics-api:v1.0 .

# Run container
docker run -p 8000:8000 nfl-analytics-api:v1.0

# Test
curl http://localhost:8000/health
```

---

## ☁️ Step 3: Deploy to Cloud

### Option A: Google Cloud Run

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/nfl-analytics-api

# Deploy to Cloud Run
gcloud run deploy nfl-analytics-api \
  --image gcr.io/YOUR_PROJECT_ID/nfl-analytics-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10 \
  --min-instances 1

# Get URL
gcloud run services describe nfl-analytics-api --region us-central1 --format 'value(status.url)'
```

### Option B: AWS Lambda + API Gateway

```python
# lambda_handler.py
import json
import torch

model = torch.jit.load("model.pt")

def lambda_handler(event, context):
    """AWS Lambda handler"""
    try:
        body = json.loads(event['body'])
        
        # Run inference
        with torch.no_grad():
            predictions = model(...)
        
        return {
            'statusCode': 200,
            'body': json.dumps({'predictions': predictions})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

Deploy:
```bash
# Package
zip -r function.zip lambda_handler.py model.pt

# Upload to S3
aws s3 cp function.zip s3://your-bucket/

# Create Lambda function
aws lambda create-function \
  --function-name nfl-analytics \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --handler lambda_handler.lambda_handler \
  --code S3Bucket=your-bucket,S3Key=function.zip \
  --memory-size 2048 \
  --timeout 30
```

---

## 📊 Step 4: Monitoring & Logging

### Prometheus Metrics

```python
# Add to inference_api.py
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
inference_time = Histogram('inference_duration_seconds', 'Inference time')

@app.post("/predict")
@inference_time.time()
async def predict(request: PredictionRequest):
    prediction_counter.inc()
    # ... existing code ...

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

---

## 🔄 Step 5: CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t nfl-analytics:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push nfl-analytics:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v0
        with:
          service: nfl-analytics-api
          image: gcr.io/PROJECT/nfl-analytics:${{ github.sha }}
          region: us-central1
```

---

## 🎯 Performance Optimization

### 1. Model Optimization

```python
# Quantization (INT8)
import torch.quantization

model_fp32 = torch.jit.load("model.pt")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
torch.jit.save(model_int8, "model_quantized.pt")

# Result: 4x smaller, 2-3x faster
```

### 2. Batch Inference

```python
@app.post("/predict_batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Process multiple requests in one batch"""
    # Combine into single batch
    # Run inference once
    # Split results
    return predictions
```

### 3. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_cached(player_state_hash):
    """Cache frequent predictions"""
    return model(...)
```

---

## 📈 Scaling Strategies

### Horizontal Scaling

```yaml
# kubernetes.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nfl-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nfl-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancing

```nginx
# nginx.conf
upstream nfl_api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://nfl_api;
    }
}
```

---

## 🔒 Security Best Practices

### 1. API Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(
    request: PredictionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401)
    # ... prediction logic ...
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request, ...):
    # ... prediction logic ...
```

### 3. Input Validation

```python
class PlayerState(BaseModel):
    x: float = Field(..., ge=0, le=120)  # Field bounds
    y: float = Field(..., ge=0, le=53.3)
    s: float = Field(..., ge=0, le=25)
    
    @validator('x', 'y')
    def check_bounds(cls, v, field):
        if field.name == 'x' and not (0 <= v <= 120):
            raise ValueError('X must be 0-120 yards')
        return v
```

---

## ✅ Deployment Checklist

- [ ] Model exported (TorchScript/ONNX)
- [ ] API tested locally
- [ ] Docker image built and tested
- [ ] Environment variables configured
- [ ] Monitoring setup (Prometheus/Grafana)
- [ ] Logging configured
- [ ] Rate limiting enabled
- [ ] Authentication implemented
- [ ] Load testing completed
- [ ] CI/CD pipeline configured
- [ ] Documentation updated
- [ ] Rollback plan prepared

---

## 📞 Production Support

**Monitoring Dashboards:**
- Grafana: http://your-domain:3000
- Prometheus: http://your-domain:9090

**Logs:**
```bash
# Cloud Run
gcloud logging read "resource.type=cloud_run_revision"

# Docker
docker logs -f container_id

# Kubernetes
kubectl logs -f deployment/nfl-api
```

---

**🚀 Your model is now production-ready!**

*Last Updated: December 2025*
