from fastapi import APIRouter, HTTPException
from services.noise_service import NoiseService

router = APIRouter()
noise_service = NoiseService()

@router.get("/")
async def get_available_noise_types():
    """Get available noise types."""
    try:
        noise_types = noise_service.get_available_noise_types()
        return {"noise_types": noise_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting noise types: {str(e)}")

@router.get("/{noise_type}")
async def get_noise_type_details(noise_type: str):
    """Get noise type details."""
    try:
        noise_details = noise_service.get_noise_type_details(noise_type)
        if noise_details is None:
            raise HTTPException(status_code=404, detail=f"Noise type {noise_type} not found")
        return noise_details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting noise type details: {str(e)}")
