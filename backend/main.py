from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import mlflow
from datetime import datetime

# Добавляем директорию backend в sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from backend.api.routes import datasets, models, noise, training, results
from core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="ML Noise Impact Analysis API",
    description="API for analyzing the impact of noise in annotations on machine learning model training quality",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(noise.router, prefix="/noise", tags=["noise"])
app.include_router(training.router, prefix="/training", tags=["training"])
app.include_router(results.router, prefix="/experiments", tags=["experiments"])


# System statistics endpoint
@app.get("/statistics", tags=["system"])
async def get_statistics():
    """Get system statistics."""
    try:
        # Get dataset count
        dataset_count = (
            len(os.listdir(settings.DATA_DIR))
            if os.path.exists(settings.DATA_DIR)
            else 0
        )

        # Get model count (hardcoded for now)
        model_count = len(settings.AVAILABLE_MODELS)

        # Get experiment count from MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        experiment_count = len(mlflow.search_experiments())

        return {
            "dataset_count": dataset_count,
            "model_count": model_count,
            "experiment_count": experiment_count,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting statistics: {str(e)}"
        )


# Health check endpoint
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
