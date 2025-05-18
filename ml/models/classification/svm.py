from typing import Dict, Any
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from ml.models.base import BaseModel

class SVMModel(BaseModel):
    """Support Vector Machine model."""
    
    def build(self):
        """Build the model."""
        return SVC(**self.params, probability=True)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model."""
        # Make predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        # Calculate ROC curve and AUC for binary classification
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y, y_prob)
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
            metrics['auc'] = auc(fpr, tpr)
        
        return metrics
