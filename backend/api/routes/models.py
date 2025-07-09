from fastapi import APIRouter, HTTPException
from services.model_service import ModelService

router = APIRouter()
model_service = ModelService()

@router.get("/")
async def get_available_models():
    """Get available models."""
    try:
        models = model_service.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

@router.get("/{model_name}")
async def get_model_details(model_name: str):
    """Get model details."""
    try:
        model_details = model_service.get_model_details(model_name)
        if model_details is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        return model_details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model details: {str(e)}")
