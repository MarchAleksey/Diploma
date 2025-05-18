from typing import List, Dict, Any, Optional

from backend.core.config import settings
from ml.models.classification.logistic_regression import LogisticRegressionModel
from ml.models.classification.random_forest import RandomForestModel
from ml.models.classification.svm import SVMModel

class ModelService:
    """Service for handling models."""
    
    def __init__(self):
        """Initialize the model service."""
        self.available_models = settings.AVAILABLE_MODELS
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models.
        
        Returns:
            List[Dict[str, Any]]: List of available models.
        """
        return self.available_models
    
    def get_model_details(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model details.
        
        Args:
            model_name (str): The model name.
        
        Returns:
            Optional[Dict[str, Any]]: Model details.
        """
        for model in self.available_models:
            if model['name'] == model_name:
                return model
        
        return None
    
    def create_model(self, model_name: str, model_params: Dict[str, Any]) -> Optional[object]:
        """Create a model.
        
        Args:
            model_name (str): The model name.
            model_params (Dict[str, Any]): Model parameters.
        
        Returns:
            Optional[object]: The created model.
        """
        if model_name == "LogisticRegression":
            return LogisticRegressionModel(model_params)
        elif model_name == "RandomForest":
            return RandomForestModel(model_params)
        elif model_name == "SVM":
            return SVMModel(model_params)
        else:
            return None
