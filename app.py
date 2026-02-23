import pickle
import numpy as np
import pandas as pd
import uvicorn
import logging
import json
import uuid
import time
import traceback
import os
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Literal, Optional
from starlette.middleware.base import BaseHTTPMiddleware

# ====================== LOGGING SETUP ======================
os.makedirs("logs", exist_ok=True)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for attr in ["request_id", "method", "url", "status_code", "duration_ms", "error_detail"]:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    fh = logging.FileHandler(log_file)
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)
    return logger

request_logger = setup_logger("request_logger", "logs/requests.log")
error_logger = setup_logger("error_logger", "logs/errors.log", logging.ERROR)
app_logger = setup_logger("app_logger", "logs/app.log")

# ====================== LOAD MODEL ======================
with open("model_artifacts/churn_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model_artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model_artifacts/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("model_artifacts/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
app_logger.info("Model artifacts loaded")

# ====================== SCHEMAS ======================
class ChurnPredictionRequest(BaseModel):
    Gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    TotalCharges: float = Field(..., ge=0)

class ChurnPredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    churn_probability: float
    no_churn_probability: float
    request_id: str

# ====================== APP ======================
app = FastAPI(title="Bank Churn Prediction API", version="2.0.0")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start_time = time.time()
        request_logger.info("Incoming request", extra={"request_id": request_id, "method": request.method, "url": str(request.url)})
        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            request_logger.info("Completed", extra={"request_id": request_id, "method": request.method, "url": str(request.url), "status_code": response.status_code, "duration_ms": duration_ms})
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            error_logger.error(f"Unhandled: {str(e)}", extra={"request_id": request_id, "error_detail": traceback.format_exc()})
            raise

app.add_middleware(LoggingMiddleware)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    errors = [{"field": " -> ".join(str(l) for l in e["loc"]), "message": e["msg"], "type": e["type"]} for e in exc.errors()]
    error_logger.error("Validation error", extra={"request_id": request_id, "error_detail": str(errors)})
    return JSONResponse(status_code=422, content={"error": "Validation Error", "detail": errors, "request_id": request_id, "timestamp": datetime.utcnow().isoformat()})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    error_logger.error(f"HTTP {exc.status_code}", extra={"request_id": request_id, "error_detail": exc.detail})
    return JSONResponse(status_code=exc.status_code, content={"error": f"HTTP {exc.status_code}", "detail": exc.detail, "request_id": request_id, "timestamp": datetime.utcnow().isoformat()})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    error_logger.error(f"Internal: {str(exc)}", extra={"request_id": request_id, "error_detail": traceback.format_exc()})
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": "Unexpected error", "request_id": request_id, "timestamp": datetime.utcnow().isoformat()})

@app.get("/")
def root():
    return {"message": "Bank Churn Prediction API v2.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request_data: ChurnPredictionRequest, request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    try:
        input_data = pd.DataFrame([request_data.model_dump()])
        for col in label_encoders:
            if col in input_data.columns:
                input_data[col] = label_encoders[col].transform(input_data[col])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        result = ChurnPredictionResponse(
            prediction=int(prediction),
            prediction_label="Churn" if prediction == 1 else "No Churn",
            churn_probability=round(float(probabilities[1]), 4),
            no_churn_probability=round(float(probabilities[0]), 4),
            request_id=request_id
        )
        app_logger.info(f"Prediction: {result.prediction_label}", extra={"request_id": request_id})
        return result
    except Exception as e:
        error_logger.error(f"Prediction error: {str(e)}", extra={"request_id": request_id, "error_detail": traceback.format_exc()})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
