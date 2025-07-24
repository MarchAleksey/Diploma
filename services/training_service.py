import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64
from typing import Dict

from backend.core.config import settings
from services.dataset_service import DatasetService
from services.model_service import ModelService
from services.noise_service import NoiseService
from ml.tracking.mlflow_logger import MLflowLogger


class TrainingService:
    """Service for handling model training."""

    def __init__(self):
        """Initialize the training service."""
        self.dataset_service = DatasetService()
        self.model_service = ModelService()
        self.noise_service = NoiseService()
        self.results_dir = settings.RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize MLflow logger
        self.mlflow_logger = MLflowLogger(
            experiment_name=settings.MLFLOW_EXPERIMENT_NAME,
            tracking_uri=settings.MLFLOW_TRACKING_URI,
        )

        # Training status
        self.training_status = {}

    def train_model(
        self,
        experiment_id: str,
        dataset_id: str,
        model_name: str,
        model_type: str,
        model_params: Dict[str, Any],
        noise_type: str,
        noise_params: Dict[str, Any],
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Train a model.

        Args:
            experiment_id (str): The experiment ID.
            dataset_id (str): The dataset ID.
            model_name (str): The model name.
            model_type (str): The model type.
            model_params (Dict[str, Any]): Model parameters.
            noise_type (str): The noise type.
            noise_params (Dict[str, Any]): Noise parameters.
            test_size (float, optional): Test size. Defaults to 0.2.
            random_state (int, optional): Random state. Defaults to 42.
            cv_folds (int, optional): Number of cross-validation folds. Defaults to 5.
            experiment_name (Optional[str], optional): Experiment name. Defaults to None.
        """
        try:
            # Update training status
            self.training_status[experiment_id] = {
                "status": "running",
                "progress": 0,
                "message": "Loading dataset",
            }

            # Load dataset
            X, y, feature_names = self.dataset_service.load_dataset(dataset_id)

            # Update training status
            self.training_status[experiment_id]["progress"] = 10
            self.training_status[experiment_id]["message"] = "Creating model"

            # Create model
            model = self.model_service.create_model(model_name, model_params)

            # Update training status
            self.training_status[experiment_id]["progress"] = 20
            self.training_status[experiment_id]["message"] = "Creating noise generator"

            # Create noise generator
            noise_generator = self.noise_service.create_noise_generator(
                noise_type, noise_params
            )

            # Update training status
            self.training_status[experiment_id]["progress"] = 30
            self.training_status[experiment_id]["message"] = "Applying noise"

            # Apply noise
            if noise_type == "label_flip":
                y_noisy = noise_generator.apply(y)
                X_noisy = X.copy()
            else:
                X_noisy = noise_generator.apply(X)
                y_noisy = y.copy()

            # Update training status
            self.training_status[experiment_id]["progress"] = 40
            self.training_status[experiment_id]["message"] = "Training model with noise"

            # Start MLflow run
            self.mlflow_logger.start_run(
                run_name=experiment_name or f"{model_name}_{noise_type}"
            )
            run = self.mlflow_logger.start_run(
                run_name=experiment_name or f"{model_name}_{noise_type}"
            )
            run_id = run.info.run_id

            # Log parameters
            self.mlflow_logger.log_params(
                {
                    "dataset_id": dataset_id,
                    "model_name": model_name,
                    "model_type": model_type,
                    "noise_type": noise_type,
                    "test_size": test_size,
                    "random_state": random_state,
                    "cv_folds": cv_folds,
                    **model_params,
                    **noise_params,
                }
            )

            # Train model with noise
            model_noisy, metrics_noisy = model.train(
                X_noisy, y_noisy, test_size=test_size, random_state=random_state
            )

            # Update training status
            self.training_status[experiment_id]["progress"] = 60
            self.training_status[experiment_id][
                "message"
            ] = "Training model without noise"

            # Train model without noise (baseline)
            model_baseline, metrics_baseline = model.train(
                X, y, test_size=test_size, random_state=random_state
            )

            # Update training status
            self.training_status[experiment_id]["progress"] = 80
            self.training_status[experiment_id]["message"] = "Generating visualizations"

            # Generate visualizations
            visualizations = {}

            # Confusion matrix
            visualizations["confusion_matrix"] = (
                self.mlflow_logger.log_confusion_matrix(y, model_noisy.predict(X))
            )

            # ROC curve (for binary classification)
            if model_type == "classification" and len(np.unique(y)) == 2:
                visualizations["roc_curve"] = self.mlflow_logger.log_roc_curve(
                    y, model_noisy.predict_proba(X)[:, 1]
                )

            # Learning curve
            visualizations["learning_curve"] = self.mlflow_logger.log_learning_curve(
                model_noisy, X, y
            )

            # Feature importance (for models that support it)
            if hasattr(model_noisy, "feature_importances_"):
                visualizations["feature_importance"] = (
                    self.mlflow_logger.log_feature_importance(
                        model_noisy.feature_importances_, feature_names
                    )
                )

            # Log metrics
            self.mlflow_logger.log_metrics(metrics_noisy)

            # Log model
            self.mlflow_logger.log_model(model_noisy, model_name)

            # End MLflow run
            self.mlflow_logger.end_run()

            # Save results
            results = {
                "experiment_id": experiment_id,
                "dataset_id": dataset_id,
                "model_name": model_name,
                "model_type": model_type,
                "model_params": model_params,
                "noise_type": noise_type,
                "noise_params": noise_params,
                "test_size": test_size,
                "random_state": random_state,
                "cv_folds": cv_folds,
                "experiment_name": experiment_name or f"{model_name}_{noise_type}",
                "metrics_noisy": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics_noisy.items()
                },
                "metrics_baseline": metrics_baseline,
                "mlflow_run_id": run_id,
                "visualizations": visualizations,
                "timestamp": datetime.now().isoformat(),
            }

            # Save results to file
            results_path = os.path.join(self.results_dir, f"{experiment_id}.json")
            with open(results_path, "w") as f:
                json.dump(results, f)

            # Update training status
            self.training_status[experiment_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Training completed successfully",
            }

        except Exception as e:
            # Update training status
            self.training_status[experiment_id] = {
                "status": "failed",
                "progress": 100,
                "message": f"Training failed: {str(e)}",
            }

            # End MLflow run if it was started
            try:
                self.mlflow_logger.end_run()
            except:
                pass

    def get_training_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get training status.

        Args:
            experiment_id (str): The experiment ID.

        Returns:
            Optional[Dict[str, Any]]: Training status.
        """
        return self.training_status.get(experiment_id)

    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get experiments.

        Returns:
            List[Dict[str, Any]]: List of experiments.
        """
        experiments = []

        # Get all files in the results directory
        files = os.listdir(self.results_dir)

        # Filter out non-JSON files
        result_files = [f for f in files if f.endswith(".json")]

        # Load experiments
        for file in result_files:
            experiment_id = file.split(".")[0]

            try:
                # Load experiment
                with open(os.path.join(self.results_dir, file), "r") as f:
                    experiment = json.load(f)

                # Add experiment to list
                experiments.append(
                    {
                        "id": experiment_id,
                        "name": experiment.get("experiment_name", ""),
                        "model": experiment.get("model_name", ""),
                        "noise_type": experiment.get("noise_type", ""),
                        "timestamp": experiment.get("timestamp", ""),
                    }
                )

            except Exception as e:
                # Skip invalid experiments
                continue

        # Sort experiments by timestamp (newest first)
        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return experiments

    def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment results.

        Args:
            experiment_id (str): The experiment ID.

        Returns:
            Optional[Dict[str, Any]]: Experiment results.
        """
        # Check if experiment exists
        results_path = os.path.join(self.results_dir, f"{experiment_id}.json")
        if not os.path.exists(results_path):
            return None

        # Load experiment
        with open(results_path, "r") as f:
            experiment = json.load(f)

        # Get dataset info
        try:
            dataset_info = self.dataset_service.get_dataset_info(
                experiment["dataset_id"]
            )
            dataset_name = dataset_info["name"]
        except:
            dataset_name = experiment["dataset_id"]

        # Get noise level
        if experiment["noise_type"] == "label_flip":
            noise_level = (
                f"{experiment['noise_params'].get('flip_ratio', 0) * 100:.1f}%"
            )
        elif experiment["noise_type"] == "gaussian_noise":
            noise_level = f"μ={experiment['noise_params'].get('mean', 0)}, σ={experiment['noise_params'].get('std', 0)}"
        elif experiment["noise_type"] == "missing_values":
            noise_level = (
                f"{experiment['noise_params'].get('missing_ratio', 0) * 100:.1f}%"
            )
        else:
            noise_level = "Unknown"

        # Get metrics
        metrics = experiment["metrics_noisy"]
        baseline_metrics = experiment["metrics_baseline"]

        # Get mlflow run ID
        mlflow_run_id = experiment.get("mlflow_run_id", "")

        # Generate visualizations
        visualizations = {}

        BASE_URL = "http://localhost:8000"

        # Load visualizations from MLflow
        for viz_name in [
            "confusion_matrix",
            "roc_curve",
            "learning_curve",
            "feature_importance",
        ]:
            if viz_name in experiment["visualizations"]:
                visualizations[viz_name] = (
                    f"{BASE_URL}/api/visualizations/{mlflow_run_id}/{viz_name}"
                )

        # Generate noise impact analysis
        noise_impact = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            if metric in metrics and metric in baseline_metrics:
                noise_impact[metric] = metrics[metric] - baseline_metrics[metric]

        noise_impact_analysis = f"""
        The introduction of {experiment['noise_type']} with level {noise_level} had the following impact on model performance:
        
        - Accuracy: {noise_impact.get('accuracy', 0) * 100:.2f}% change
        - Precision: {noise_impact.get('precision', 0) * 100:.2f}% change
        - Recall: {noise_impact.get('recall', 0) * 100:.2f}% change
        - F1 Score: {noise_impact.get('f1', 0) * 100:.2f}% change
        
        Overall, the noise {
            'significantly degraded' if sum(noise_impact.values()) < -0.1 else
            'slightly degraded' if sum(noise_impact.values()) < 0 else
            'had minimal impact on' if abs(sum(noise_impact.values())) < 0.05 else
            'slightly improved' if sum(noise_impact.values()) > 0 else
            'significantly improved'
        } model performance.
        """

        # Generate dummy PDF and CSV for download
        report_pdf = base64.b64encode(b"Dummy PDF report").decode("utf-8")
        results_csv = base64.b64encode(b"Dummy CSV data").decode("utf-8")

        # Return formatted results
        return {
            "name": experiment.get("experiment_name", ""),
            "dataset": dataset_name,
            "model": experiment.get("model_name", ""),
            "noise_type": experiment.get("noise_type", ""),
            "noise_level": noise_level,
            "date": datetime.fromisoformat(experiment.get("timestamp", "")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "metrics": metrics,
            "baseline_comparison": baseline_metrics,
            "visualizations": visualizations,
            "noise_impact_analysis": noise_impact_analysis,
            "report_pdf": report_pdf,
            "results_csv": results_csv,
        }
