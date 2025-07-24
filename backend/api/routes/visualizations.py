from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from mlflow.tracking import MlflowClient
import os

router = APIRouter()
client = MlflowClient()

@router.get("/visualizations/{run_id}/{viz_type}")
async def get_visualization(run_id: str, viz_type: str):
    # Определяем пути к артефактам
    viz_map = {
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "learning_curve": "learning_curve.png",
        "feature_importance": "feature_importance.png"
    }
    
    if viz_type not in viz_map:
        raise HTTPException(404, "Visualization type not found")
    
    artifact_paths = f"artifacts/{viz_map[viz_type]}"

    try:
        #print(f"Request received: run_id={run_id}, viz_type={viz_type}")
        # Создаем временную директорию
        temp_dir = f"/tmp/mlflow/{run_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Скачиваем артефакт
        local_path = client.download_artifacts(run_id, artifact_paths[viz_type], temp_dir)
        
        # Проверяем существование файла
        if not os.path.exists(local_path):
            raise HTTPException(404, "File not found after download")
        
        return FileResponse(local_path, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(500, f"Error loading visualization: {str(e)}")