from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid

from services.training_service import TrainingService

router = APIRouter()
training_service = TrainingService()


class TrainingConfig(BaseModel):
    """Training configuration."""

    dataset_id: str
    model_name: str
    model_type: str
    model_params: Dict[str, Any]
    noise_type: str
    noise_params: Dict[str, Any]
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    experiment_name: Optional[str] = None


@router.post("/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start model training."""
    try:
        # Generate experiment ID
        experiment_id = str(uuid.uuid4())

        # Start training in the background
        background_tasks.add_task(
            training_service.train_model,
            experiment_id=experiment_id,
            dataset_id=config.dataset_id,
            model_name=config.model_name,
            model_type=config.model_type,
            model_params=config.model_params,
            noise_type=config.noise_type,
            noise_params=config.noise_params,
            test_size=config.test_size,
            random_state=config.random_state,
            cv_folds=config.cv_folds,
            experiment_name=config.experiment_name
            or f"{config.model_name}_{config.noise_type}",
        )

        return {
            "success": True,
            "experiment_id": experiment_id,
            "message": "Training started successfully",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting training: {str(e)}"
        )


@router.get("/{experiment_id}/status")
async def get_training_status(experiment_id: str):
    """Get training status."""
    try:
        status = training_service.get_training_status(experiment_id)
        if status is None:
            raise HTTPException(
                status_code=404, detail=f"Experiment {experiment_id} not found"
            )
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting training status: {str(e)}"
        )
