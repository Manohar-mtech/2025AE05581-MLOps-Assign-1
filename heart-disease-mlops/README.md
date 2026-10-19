# Heart Disease Risk – End-to-End MLOps Pipeline

Predicts the presence of heart disease from patient health data (UCI Heart Disease, Cleveland subset)
and serves the model as a containerised, monitored REST API.

> Repository: `<add your GitHub URL here>`

## Architecture

```
 UCI dataset ──► src/data.py (clean) ──► src/eda.py ──► reports/figures
                      │
                      ▼
          src/train.py  (sklearn Pipeline: impute → scale/one-hot → model)
          GridSearchCV + stratified 5-fold CV ──► MLflow (params, metrics, plots, model)
                      │
                      ▼
              models/model.joblib + metadata.json
                      │
   GitHub Actions: flake8 → pytest → EDA → train → docker build → smoke test
                      │
                      ▼
       Docker image (FastAPI + uvicorn) ──► Kubernetes (Deployment + LoadBalancer Service)
                      │ /metrics
                      ▼
               Prometheus ──► Grafana dashboard
```

## Repository layout

| Path | Purpose |
|---|---|
| `data/raw/`, `data/processed/` | Raw UCI file and the cleaned CSV (`heart_clean.csv`) |
| `scripts/download_data.py` | Downloads the dataset from UCI |
| `src/` | `data.py`, `features.py` (preprocessing), `modeling.py`, `train.py` (MLflow), `eda.py`, `predict.py` |
| `api/` | FastAPI app (`app.py`) and pydantic schemas |
| `tests/` | Pytest unit + API tests |
| `Dockerfile`, `docker-compose.yml` | Serving image; compose adds Prometheus + Grafana |
| `k8s/` | Deployment, Service (LoadBalancer), optional Ingress, monitoring stack |
| `monitoring/` | Prometheus config + provisioned Grafana dashboard |
| `.github/workflows/ci.yml` | CI pipeline |
| `reports/figures/` | EDA plots |
| `screenshots/` | Evidence for the report |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # Python 3.11 recommended
python scripts/download_data.py                        # skips if data/raw already populated
```

## Run the pipeline

```bash
python -m src.data          # cleaned dataset -> data/processed/heart_clean.csv
python -m src.eda           # figures -> reports/figures/
python -m src.train         # tunes 3 models, logs to MLflow, saves models/model.joblib
mlflow ui --backend-store-uri ./mlruns                 # http://127.0.0.1:5000
pytest                                                 # unit tests
flake8 src api tests scripts                           # lint
python -m src.predict sample_input.json                # offline inference
```

## Serve locally

```bash
uvicorn api.app:app --reload                           # http://127.0.0.1:8000/docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
```

Response:
```json
{"prediction": 0, "label": "No heart disease", "confidence": 0.80, "probability_disease": 0.20}
```
Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `GET /metrics` (Prometheus).

## Docker

```bash
python -m src.train                                    # the image bundles models/model.joblib
docker build -t heart-disease-api:latest .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:latest
scripts/smoke_test.sh http://localhost:8000
docker logs heart-api                                  # request logs
```

## Kubernetes (Minikube or Docker Desktop)

```bash
minikube start
eval $(minikube docker-env)                            # build straight into Minikube's Docker
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
minikube tunnel                                        # separate terminal -> EXTERNAL-IP for the LoadBalancer
scripts/smoke_test.sh http://<EXTERNAL-IP>
```
Docker Desktop Kubernetes: skip the Minikube lines; the Service is reachable on `http://localhost`.
For GKE/EKS/AKS push the image to a registry and change `image:` in `k8s/deployment.yaml`.
Optional Ingress: `minikube addons enable ingress && kubectl apply -f k8s/ingress.yaml`.

## Monitoring & logging

* Every request is logged (method, path, status, latency, client) plus each prediction and its confidence.
* Prometheus metrics: `api_requests_total`, `api_request_latency_seconds`, `api_predictions_total`,
  `api_prediction_confidence`.
* Local stack: `docker compose up --build` → API :8000, Prometheus :9090, Grafana :3000 (admin/admin,
  dashboard "Heart Disease API" is pre-provisioned). Generate traffic with `scripts/smoke_test.sh` in a loop.
* Kubernetes: `kubectl apply -f k8s/monitoring.yaml`; in Grafana import `monitoring/grafana/dashboards/heart-api.json`.

## Modelling choices (summary – expand in your report)

* **Target:** original `num` (0–4) → binary (0 = no disease, 1 = disease).
* **Cleaning:** header names added, `?` → NaN (4 in `ca`, 2 in `thal`), duplicates dropped, floats cast.
  Imputation (median / most-frequent) lives **inside** the sklearn Pipeline so it is fitted on training
  folds only and is applied identically at inference.
* **Features:** `age, trestbps, chol, thalach, oldpeak` standardised; `sex, cp, fbs, restecg, exang, slope, ca, thal`
  one-hot encoded.
* **Models:** Logistic Regression, Random Forest, Gradient Boosting; each grid-searched on ROC-AUC with
  stratified 5-fold CV on an 80 % training split. The best family is chosen by CV ROC-AUC (the 20 % hold-out
  test set is only used for reporting), and saved as the deployment artifact.
* **Metrics:** accuracy, precision, recall, F1, ROC-AUC (CV mean ± std and hold-out).

## Experiment tracking

One parent MLflow run (`training-session`) with a nested run per model family. Logged: tuned parameters,
grid size, CV and test metrics, grid-search results CSV, confusion matrix, ROC curve, feature-importance
plot, the fitted pipeline (with signature + input example), and the final `model.joblib` + `metadata.json`.

## CI/CD

`.github/workflows/ci.yml` runs on push/PR: **flake8 → pytest (with coverage) → EDA → training → upload
artifacts (reports, mlruns, model) → Docker build → container smoke test**. Any lint, test, training or
smoke-test failure fails the workflow and the logs show the failing step.

## Screenshots to capture for the report (`screenshots/`)

EDA plots · MLflow experiment list and run comparison · `pytest` output · GitHub Actions green run
(+ one deliberate failure) · `docker build` / `docker run` / smoke test · `kubectl get pods,svc` ·
`/predict` response from the cluster · Prometheus targets page · Grafana dashboard.
