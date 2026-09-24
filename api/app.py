"""
FastAPI Salary Prediction API
Simple REST API to make predictions
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import sys
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "jobs_raw.csv"


# ============================================================
# LOAD TRAINED FILES
# ============================================================

print("Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

from src.preprocessing import FeatureEngineer

engineer = FeatureEngineer.load(str(PREPROCESSOR_PATH))

# Load raw data for job-title popularity
raw_data = pd.read_csv(RAW_DATA_PATH)

print("✅ All models loaded successfully!")


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(
    title="💼 Salary Predictor API",
    description="Predict job salary in India",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST FORMAT
# ============================================================

class SalaryRequest(BaseModel):
    job_title: str
    experience: int
    location: str
    company: str
    company_size: str
    industry: str
    education: str
    skills: list[str]
    posted_date: str


# ============================================================
# RESPONSE FORMAT
# ============================================================

class SalaryResponse(BaseModel):
    predicted_salary: float
    salary_range_min: float
    salary_range_max: float
    confidence: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health_check():
    return {
        "status": "API Running ✅",
        "model": "Gradient Boosting",
        "version": "1.0"
    }


# ============================================================
# PREDICT SALARY
# ============================================================

@app.post("/predict", response_model=SalaryResponse)
def predict_salary(request: SalaryRequest):

    # --------------------------------------------------------
    # Create dataframe from user input
    # --------------------------------------------------------

    df = pd.DataFrame([{
        "job_title": request.job_title,
        "experience_required": request.experience,
        "location": request.location,
        "company_size": request.company_size,
        "skills": " ".join(request.skills),
        "company": request.company,
        "industry": request.industry,
        "education": request.education,
        "posted_date": request.posted_date,
        "applications": 100,
        "views": 1000,
        "salary": 0
    }])


    # --------------------------------------------------------
    # Handle unknown categorical values
    # --------------------------------------------------------

    for col in ["job_title", "company", "industry", "education"]:

        encoder = engineer.label_encoders[col]

        if df[col].iloc[0] not in encoder.classes_:
            df.loc[0, col] = encoder.classes_[0]


    # --------------------------------------------------------
    # Handle unknown location
    # --------------------------------------------------------

    if request.location not in engineer.location_salary_map:

        first_location = next(
            iter(engineer.location_salary_map)
        )

        df.loc[0, "location"] = first_location


    # --------------------------------------------------------
    # Transform features
    # --------------------------------------------------------

    df_transformed = engineer.transform(df)

    feature_cols = engineer.get_feature_columns()

    X = df_transformed[feature_cols].copy()


    # --------------------------------------------------------
    # Job title popularity
    # --------------------------------------------------------

    job_title_counts = raw_data["job_title"].value_counts()

    original_job_title = request.job_title

    if original_job_title in job_title_counts:
        X.loc[:, "job_title_popularity"] = job_title_counts[original_job_title]
    else:
        X.loc[:, "job_title_popularity"] = 1


    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    X = X.fillna(0)


    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    X_scaled = scaler.transform(X)


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(X_scaled)[0]


    # --------------------------------------------------------
    # Salary range
    # --------------------------------------------------------

    min_salary = round(prediction * 0.85, 2)
    max_salary = round(prediction * 1.15, 2)


    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if request.experience >= 5:
        confidence = "High"
    elif request.experience >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return SalaryResponse(
        predicted_salary=round(float(prediction), 2),
        salary_range_min=min_salary,
        salary_range_max=max_salary,
        confidence=confidence
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print("\n" + "=" * 50)
    print("🚀 Starting API Server...")
    print("=" * 50)
    print("📍 URL: http://127.0.0.1:8000")
    print("📚 Docs: http://127.0.0.1:8000/docs")
    print("=" * 50 + "\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )