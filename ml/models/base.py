from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the model.
        
        Args:
            params (Dict[str, Any], optional): Model parameters.
        """
        self.params = params or {}
        self.model = None
    
    @abstractmethod
    def build(self) -> BaseEstimator:
        """Build the model.
        
        Returns:
            BaseEstimator: The built model.
        """
        pass
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Train the model.
        
        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Target.
            test_size (float, optional): Test size. Defaults to 0.2.
            random_state (int, optional): Random state. Defaults to 42.
        
        Returns:
            Tuple[BaseEstimator, Dict[str, Any]]: The trained model and metrics.
        """
        # Build the model
        self.model = self.build()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        metrics = self.evaluate(X_test, y_test)
        
        return self.model, metrics
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model.
        
        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Target.
        
        Returns:
            Dict[str, Any]: Evaluation metrics.
        """
        pass
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'accuracy',
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Cross-validate the model.
        
        Args:
            X (np.ndarray): Features.
            y (np.ndarray): Target.
            cv (int, optional): Number of folds. Defaults to 5.
            scoring (str, optional): Scoring metric. Defaults to 'accuracy'.
            random_state (int, optional): Random state. Defaults to 42.
        
        Returns:
            Dict[str, Any]: Cross-validation results.
        """
        # Build the model
        self.model = self.build()
        
        # Cross-validate
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, scoring=scoring
        )
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X (np.ndarray): Features.
        
        Returns:
            np.ndarray: Predictions.
        
        Raises:
            ValueError: If the model is not trained.
        """
        if self.model is None:
            raise ValueError("Model is not trained")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save the model.
        
        Args:
            path (str): Path to save the model.
        
        Raises:
            ValueError: If the model is not trained.
        """
        if self.model is None:
            raise ValueError("Model is not trained")
        
        import joblib
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load the model.
        
        Args:
            path (str): Path to load the model from.
        
        Returns:
            BaseModel: The loaded model.
        """
        import joblib
        model = cls()
        model.model = joblib.load(path)
        return model
