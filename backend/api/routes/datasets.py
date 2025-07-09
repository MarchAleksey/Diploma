from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from services.dataset_service import DatasetService

router = APIRouter()
dataset_service = DatasetService()

@router.get("/")
async def get_available_datasets():
    """Get available datasets."""
    try:
        datasets = dataset_service.get_available_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting datasets: {str(e)}")

@router.get("/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    """Get dataset information."""
    try:
        dataset_info = dataset_service.get_dataset_info(dataset_id)
        return dataset_info
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """Upload a dataset."""
    try:
        # Read the dataset
        content = await file.read()
        
        # Save the dataset
        dataset_id = dataset_service.save_dataset(content, file.filename, target_column)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "message": "Dataset uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}")
