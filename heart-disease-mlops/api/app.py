"""FastAPI model-serving app with request logging and Prometheus metrics."""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)

from api.schemas import PatientFeatures, PredictionResponse
from src.predict import load_model, predict_records

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["path"])
PREDICTIONS = Counter("api_predictions_total", "Predictions served", ["label"])
CONFIDENCE = Histogram("api_prediction_confidence", "Confidence of served predictions",
                       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", "models/model.joblib")
    app.state.model = None
    app.state.metadata = {}
    try:
        app.state.model = load_model(path)
        meta = Path(path).with_name("metadata.json")
        if meta.exists():
            app.state.metadata = json.loads(meta.read_text())
        log.info("Model loaded from %s", path)
    except Exception:
        log.exception("Could not load model from %s", path)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"
        if path != "/metrics":
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            log.info("request method=%s path=%s status=%s latency_ms=%.1f client=%s",
                     request.method, request.url.path, status, elapsed * 1000,
                     request.client.host if request.client else "-")


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded}


@app.get("/model-info")
def model_info(request: Request):
    return request.app.state.metadata or {"detail": "no metadata available"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predict_records(model, [payload.model_dump()])[0]
    except Exception:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["confidence"])
    log.info("prediction=%s confidence=%.3f", result["prediction"], result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
