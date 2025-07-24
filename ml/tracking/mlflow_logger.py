import os
import tempfile
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from mlflow.entities import Run
from typing import Dict, Any, Optional
from sklearn.base import BaseEstimator
import io
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import learning_curve


class MLflowLogger:
    """Class for logging experiments to MLflow."""

    def __init__(self, experiment_name, tracking_uri=None):
        """Initialize the MLflow logger.

        Args:
            experiment_name (str): Name of the experiment.
            tracking_uri (Optional[str], optional): MLflow tracking URI.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.active_run = None

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(
                experiment_name, 
                artifact_location=os.path.abspath("./mlruns")
            )
        else:
            self.experiment_id = self.experiment.experiment_id

        mlflow.set_experiment(experiment_name)

    def _save_visualization(self, plt, filename: str) -> str:
        """Internal method to save visualization reliably."""
        artifact_dir = "artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, filename)
            plt.savefig(file_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Log the artifact
            mlflow.log_artifact(file_path, artifact_path=artifact_dir)
            
            # Verify the artifact was saved
            if self.active_run:
                artifacts = mlflow.tracking.MlflowClient().list_artifacts(
                    self.active_run.info.run_id, 
                    artifact_dir
                )
                if not any(a.path == f"{artifact_dir}/{filename}" for a in artifacts):
                    raise RuntimeError(f"Failed to save {filename}")
        
        return f"{artifact_dir}/{filename}"

    def start_run(self, run_name=None) -> Run:
        if self.active_run:
            self.end_run()

        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name
        )
        return self.active_run

    def end_run(self):
        """End the current run."""
        if self.active_run:
            mlflow.end_run()
            self.active_run = None

    @property
    def run(self) -> Run:
        """Get the current active run."""
        if not self.active_run:
            raise RuntimeError("No active MLflow run")
        return self.active_run

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

    def get_sklearn_model(self) -> BaseEstimator:
        """Get the underlying sklearn model.

        Returns:
            BaseEstimator: The sklearn model.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.model is None:
            raise ValueError("Model is not trained")
        return self.model

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
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Add labels
        classes = np.unique(y_true)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Add text
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        # Сохраняем во временный файл
        artifact_dir = "artifacts"
        filename = "confusion_matrix.png"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, filename)
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            
            # Логируем артефакт
            mlflow.log_artifact(file_path, artifact_path=artifact_dir)
        
        return f"{artifact_dir}/{filename}"

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
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
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
        plt.title("Learning Curve")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.legend(loc="best")

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Log figure
        path = f"learning_curve.png"
        mlflow.log_figure(plt.gcf(), path)

        plt.close()

        return path

    def log_feature_importance(
        self, feature_importance: np.ndarray, feature_names: Optional[np.ndarray] = None
    ) -> str:
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
        plt.title("Feature Importance")
        plt.bar(
            range(len(sorted_feature_importance)),
            sorted_feature_importance,
            align="center",
        )
        plt.xticks(
            range(len(sorted_feature_importance)), sorted_feature_names, rotation=90
        )
        plt.tight_layout()

        # Save figure
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Log figure
        path = f"feature_importance.png"
        mlflow.log_figure(plt.gcf(), path)

        plt.close()

        return path
