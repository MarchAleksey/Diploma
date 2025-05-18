import os
from pydantic import BaseSettings
from typing import Dict, Any, List

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML Noise Impact Analysis"
    
    # Data directories
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "..", "data", "raw")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "..", "data", "processed")
    RESULTS_DIR: str = os.path.join(BASE_DIR, "..", "data", "results")
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "noise-impact-analysis"
    
    # Available models
    AVAILABLE_MODELS: List[Dict[str, Any]] = [
        {
            "name": "LogisticRegression",
            "type": "classification",
            "description": "Logistic Regression classifier",
            "parameters": [
                {
                    "name": "C",
                    "type": "float",
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "Inverse of regularization strength"
                },
                {
                    "name": "max_iter",
                    "type": "int",
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "description": "Maximum number of iterations"
                },
                {
                    "name": "solver",
                    "type": "select",
                    "default": "lbfgs",
                    "options": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                    "description": "Algorithm to use in the optimization problem"
                }
            ]
        },
        {
            "name": "RandomForest",
            "type": "classification",
            "description": "Random Forest classifier",
            "parameters": [
                {
                    "name": "n_estimators",
                    "type": "int",
                    "default": 100,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "description": "Number of trees in the forest"
                },
                {
                    "name": "max_depth",
                    "type": "int",
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "description": "Maximum depth of the tree"
                },
                {
                    "name": "min_samples_split",
                    "type": "int",
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "description": "Minimum number of samples required to split an internal node"
                }
            ]
        },
        {
            "name": "SVM",
            "type": "classification",
            "description": "Support Vector Machine classifier",
            "parameters": [
                {
                    "name": "C",
                    "type": "float",
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "description": "Regularization parameter"
                },
                {
                    "name": "kernel",
                    "type": "select",
                    "default": "rbf",
                    "options": ["linear", "poly", "rbf", "sigmoid"],
                    "description": "Kernel type to be used in the algorithm"
                },
                {
                    "name": "gamma",
                    "type": "select",
                    "default": "scale",
                    "options": ["scale", "auto"],
                    "description": "Kernel coefficient"
                }
            ]
        },
        {
            "name": "LinearRegression",
            "type": "regression",
            "description": "Linear Regression model",
            "parameters": [
                {
                    "name": "fit_intercept",
                    "type": "bool",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model"
                },
                {
                    "name": "normalize",
                    "type": "bool",
                    "default": False,
                    "description": "Whether to normalize the data before regression"
                }
            ]
        },
        {
            "name": "RandomForestRegressor",
            "type": "regression",
            "description": "Random Forest regressor",
            "parameters": [
                {
                    "name": "n_estimators",
                    "type": "int",
                    "default": 100,
                    "min": 10,
                    "max": 500,
                    "step": 10,
                    "description": "Number of trees in the forest"
                },
                {
                    "name": "max_depth",
                    "type": "int",
                    "default": 10,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "description": "Maximum depth of the tree"
                },
                {
                    "name": "min_samples_split",
                    "type": "int",
                    "default": 2,
                    "min": 2,
                    "max": 20,
                    "step": 1,
                    "description": "Minimum number of samples required to split an internal node"
                }
            ]
        }
    ]
    
    # Available noise types
    AVAILABLE_NOISE_TYPES: List[Dict[str, Any]] = [
        {
            "name": "label_flip",
            "description": "Randomly flip labels in the dataset",
            "parameters": [
                {
                    "name": "flip_ratio",
                    "type": "float",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Ratio of labels to flip"
                },
                {
                    "name": "random_state",
                    "type": "int",
                    "default": 42,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "description": "Random state for reproducibility"
                }
            ]
        },
        {
            "name": "gaussian_noise",
            "description": "Add Gaussian noise to features",
            "parameters": [
                {
                    "name": "mean",
                    "type": "float",
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "description": "Mean of the Gaussian noise"
                },
                {
                    "name": "std",
                    "type": "float",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Standard deviation of the Gaussian noise"
                },
                {
                    "name": "random_state",
                    "type": "int",
                    "default": 42,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "description": "Random state for reproducibility"
                }
            ]
        },
        {
            "name": "missing_values",
            "description": "Introduce missing values in the dataset",
            "parameters": [
                {
                    "name": "missing_ratio",
                    "type": "float",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "description": "Ratio of values to set as missing"
                },
                {
                    "name": "random_state",
                    "type": "int",
                    "default": 42,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "description": "Random state for reproducibility"
                }
            ]
        }
    ]
    
    class Config:
        case_sensitive = True

settings = Settings()

# Create directories if they don't exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
