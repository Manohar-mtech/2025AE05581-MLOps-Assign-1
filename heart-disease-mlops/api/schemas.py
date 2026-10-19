"""Request / response models for the prediction API."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=260, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Optional[int] = Field(None, description="3 = normal, 6 = fixed defect, 7 = reversible")

    @field_validator("thal")
    @classmethod
    def _valid_thal(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}})


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
