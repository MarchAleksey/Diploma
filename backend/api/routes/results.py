from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

from services.training_service import TrainingService

router = APIRouter()
training_service = TrainingService()

@router.get("/")
async def get_experiments():
    """Get experiments."""
    try:
        experiments = training_service.get_experiments()
        return {"experiments": experiments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experiments: {str(e)}")

@router.get("/{experiment_id}")
async def get_experiment_results(experiment_id: str):
    """Get experiment results."""
    try:
        results = training_service.get_experiment_results(experiment_id)
        if results is None:
            raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting experiment results: {str(e)}")
