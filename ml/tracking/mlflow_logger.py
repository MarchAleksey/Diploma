import mlflow
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import io
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve

class MLflowLogger:
    """Class for logging experiments to MLflow."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        """Initialize the MLflow logger.
        
        Args:
            experiment_name (str): Name of the experiment.
            tracking_uri (Optional[str], optional): MLflow tracking URI.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        self.experiment = mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new run.
        
        Args:
            run_name (Optional[str], optional): Name of the run.
        """
        mlflow.start_run(run_name=run_name)
    
    def end_run(self) -> None:
        """End the current run."""
        mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to log.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics.
        
        Args:
            metrics (Dict[str, Any]): Metrics to log.
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
    
    def log_model(self, model, model_name: str) -> None:
        """Log a model.
        
        Args:
            model: Model to log.
            model_name (str): Name of the model.
        """
        mlflow.sklearn.log_model(model, model_name)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Log a confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        
        Returns:
            str: Path to the saved confusion matrix.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = np.unique(y_true)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log figure
        path = f"confusion_matrix.png"
        mlflow.log_figure(plt.gcf(), path)
        
        plt.close()
        
        return path
    
    def log_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> str:
        """Log a ROC curve.
        
        Args:
            y_true (np.ndarray): True labels.
            y_score (np.ndarray): Predicted scores.
        
        Returns:
            str: Path to the saved ROC curve.
        """
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log figure
        path = f"roc_curve.png"
        mlflow.log_figure(plt.gcf(), path)
        
        plt.close()
        
        return path
    
    def log_learning_curve(self, estimator, X: np.ndarray, y: np.ndarray) -> str:
        """Log a learning curve.
        
        Args:
            estimator: Estimator to use.
            X (np.ndarray): Features.
            y (np.ndarray): Target.
        
        Returns:
            str: Path to the saved learning curve.
        """
        # Compute learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and standard deviation
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 8))
        plt.title('Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.legend(loc='best')
        
        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log figure
        path = f"learning_curve.png"
        mlflow.log_figure(plt.gcf(), path)
        
        plt.close()
        
        return path
    
    def log_feature_importance(self, feature_importance: np.ndarray, feature_names: Optional[np.ndarray] = None) -> str:
        """Log feature importance.
        
        Args:
            feature_importance (np.ndarray): Feature importance.
            feature_names (Optional[np.ndarray], optional): Feature names.
        
        Returns:
            str: Path to the saved feature importance.
        """
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
        
        # Sort feature importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_feature_importance = feature_importance[indices]
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(sorted_feature_importance)), sorted_feature_importance, align='center')
        plt.xticks(range(len(sorted_feature_importance)), sorted_feature_names, rotation=90)
        plt.tight_layout()
        
        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log figure
        path = f"feature_importance.png"
        mlflow.log_figure(plt.gcf(), path)
        
        plt.close()
        
        return path
