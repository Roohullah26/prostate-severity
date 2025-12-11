# DEPLOYMENT GUIDE - Prostate Tumor Analyzer

## Quick Start (Choose One)

### Option 1: Start Streamlit UI (Easiest)
```bash
python deploy_start_ui.py
```
Opens interactive web interface at **http://localhost:8501**

### Option 2: Start FastAPI Server (REST API)
```bash
python deploy_start_server.py
```
API available at **http://localhost:8000/docs**

### Option 3: Manual Startup
```bash
cd deploy_clean
pip install -r requirements.txt
python -m streamlit run webapp/streamlit_app.py
```

---

## Deployment Architectures

### Local/Development Deployment
```
Your Machine
    ↓
[deploy_clean/]
    ├─ Streamlit UI (port 8501)
    ├─ FastAPI Server (port 8000)
    └─ Model: baseline_real_t2_adc_3s_ep1.pth
```

**Time to deploy:** 2 minutes  
**Requirements:** Python 3.11, venv  
**Cost:** $0

---

### Docker Deployment (Recommended for Production)
```bash
cd deploy_clean
docker build -t prostate-analyzer:latest .
docker run -p 8000:8000 -p 8501:8501 prostate-analyzer:latest
```

**Benefits:**
- Reproducible across machines
- Easy to scale
- Can push to cloud registry

**Time to deploy:** 5 minutes + Docker build time (~5-10 min)

---

### Kubernetes Deployment (Enterprise Scale)
```yaml
# deploy_clean/k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prostate-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prostate-analyzer
  template:
    metadata:
      labels:
        app: prostate-analyzer
    spec:
      containers:
      - name: api
        image: your-registry/prostate-analyzer:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: prostate-analyzer-service
spec:
  selector:
    app: prostate-analyzer
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Benefits:**
- Auto-scaling
- High availability
- Load balancing
- Rolling updates

**Time to deploy:** 10 minutes (after setup)  
**Requirements:** Kubernetes cluster

---

### Cloud Deployment Options

#### AWS - Simple
```bash
# 1. Build and push image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker build -t prostate-analyzer deploy_clean/
docker tag prostate-analyzer:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/prostate-analyzer:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/prostate-analyzer:latest

# 2. Deploy to ECS
aws ecs create-service \
  --cluster prostate-cluster \
  --service-name prostate-api \
  --task-definition prostate-analyzer:1 \
  --desired-count 2
```

#### AWS - Serverless (Lambda)
```bash
# Deploy to AWS Lambda (use Zappa or similar)
pip install zappa
cd deploy_clean
zappa init
zappa deploy production
```

**Pros:** Pay per request, auto-scaling, no server management  
**Cons:** Cold start latency (~1-3s), memory/timeout limits

#### GCP - Cloud Run
```bash
# 1. Build & push
gcloud builds submit --tag gcr.io/PROJECT_ID/prostate-analyzer deploy_clean/

# 2. Deploy
gcloud run deploy prostate-analyzer \
  --image gcr.io/PROJECT_ID/prostate-analyzer \
  --platform managed \
  --memory 2Gi \
  --port 8000 \
  --allow-unauthenticated
```

#### Azure - Container Instances
```bash
# 1. Build & push to Azure Container Registry
az acr build --registry myregistry --image prostate-analyzer:latest deploy_clean/

# 2. Deploy
az container create \
  --resource-group mygroup \
  --name prostate-api \
  --image myregistry.azurecr.io/prostate-analyzer:latest \
  --ports 8000 \
  --cpu 2 --memory 2
```

---

## Testing the Deployment

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Predict Endpoint
```bash
# Example prediction request (requires DICOM or preprocessed image)
curl -X POST http://localhost:8000/predict \
  -F "file=@your_mri_image.npy" \
  -F "modality=t2w"
```

### 3. Batch Prediction
```bash
curl -X POST http://localhost:8000/batch \
  -F "files=@image1.npy" \
  -F "files=@image2.npy"
```

### 4. View API Docs
Open **http://localhost:8000/docs** in browser

---

## Production Checklist

- [ ] Deploy locally and test
- [ ] Build Docker image
- [ ] Push to container registry (Docker Hub, ECR, GCR, etc.)
- [ ] Deploy to staging environment
- [ ] Run performance tests
- [ ] Validate with sample data
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure logging (ELK stack, CloudWatch)
- [ ] Set up CI/CD pipeline
- [ ] Document API usage
- [ ] Train team on operations
- [ ] Deploy to production
- [ ] Monitor model performance & drift

---

## Environment Variables

Set these in your deployment environment:

```bash
# GPU Support (if available)
export CUDA_VISIBLE_DEVICES=0

# Model Path (if custom location)
export MODEL_PATH=/path/to/baseline_real_t2_adc_3s_ep1.pth

# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000

# Logging
export LOG_LEVEL=INFO

# Security (for production)
export API_KEY=your-secure-key-here
export ENABLE_AUTH=true
```

---

## Monitoring & Operations

### Essential Metrics to Track
```
• Inference latency (p50, p95, p99)
• Throughput (requests/second)
• Error rate (4xx, 5xx errors)
• Model output distribution (drift detection)
• GPU utilization (if applicable)
• Memory usage
• API availability (uptime %)
```

### Recommended Tools
```
Monitoring:  Prometheus + Grafana
Logging:     ELK Stack or CloudWatch
Tracing:     Jaeger or Datadog
Alerting:    PagerDuty or OpsGenie
CI/CD:       GitHub Actions, GitLab CI, or Jenkins
```

---

## Scaling Strategy

### Vertical Scaling (Make instance bigger)
```
1 GPU → 2 GPUs
4GB RAM → 8GB RAM
2 CPU → 4 CPU
```

### Horizontal Scaling (Add more instances)
```
Single instance → Load balancer + 2-3 instances
→ Kubernetes with auto-scaling (3-10+ pods)
```

### When to Scale
- Latency > 500ms
- CPU/GPU > 80% utilized
- Request queue building up
- Users reporting slowness

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port already in use | `lsof -i :8000` to find process, then kill or use different port |
| Model not found | Verify path in `.env` or verify `models/baseline_real_t2_adc_3s_ep1.pth` exists |
| Out of memory | Reduce batch size or increase instance memory |
| Slow inference | Enable GPU, check model quantization, reduce image resolution |
| High latency | Add load balancer, scale to multiple instances, use caching |
| API errors | Check logs: `docker logs <container_id>` |

---

## Security Considerations

For production clinical use:

```
1. AUTHENTICATION
   ├─ API Key validation
   ├─ JWT tokens for web UI
   └─ OAuth2 for enterprise SSO

2. ENCRYPTION
   ├─ HTTPS/TLS for API
   ├─ Encrypt data at rest
   └─ Encrypt in transit (TLS 1.3)

3. COMPLIANCE
   ├─ HIPAA audit logging
   ├─ FDA 21 CFR Part 11 (if regulated)
   ├─ GDPR data handling
   └─ SOC2 compliance

4. MONITORING
   ├─ Intrusion detection
   ├─ Rate limiting (prevent DDoS)
   ├─ Input validation (prevent injection)
   └─ Access logging (audit trail)
```

---

## Next Steps

1. **Immediate (Today)**
   - [ ] Test deployment: `python deploy_start_server.py`
   - [ ] Verify model loads
   - [ ] Test inference with sample data

2. **This Week**
   - [ ] Build Docker image
   - [ ] Push to registry
   - [ ] Deploy to staging

3. **This Month**
   - [ ] Set up monitoring
   - [ ] Validate with real data
   - [ ] Get regulatory approval (if needed)
   - [ ] Deploy to production

---

## Support & Resources

```
Model Card:      models/baseline_real_t2_adc_3s_ep1.pth
Architecture:    src/size_predictor_model.py
API Reference:   http://localhost:8000/docs
Documentation:   DELIVERY_COMPLETE.md
```

**Status:** ✓ Ready for deployment

