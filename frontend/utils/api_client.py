import os
import requests

class APIClient:
    """Client for communicating with the FastAPI backend."""
    
    def __init__(self, base_url=None):
        """Initialize the API client.
    
        Args:
            base_url (str, optional): The base URL of the FastAPI backend.
        """
        self.base_url = base_url or os.environ.get("BACKEND_URL", "http://backend:8000")
    
    def _get(self, endpoint, params=None):
        """Make a GET request to the API.
        
        Args:
            endpoint (str): The API endpoint.
            params (dict, optional): Query parameters.
        
        Returns:
            dict: The response JSON.
        
        Raises:
            Exception: If the request fails.
        """
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def _post(self, endpoint, data=None, files=None):
        """Make a POST request to the API.
        
        Args:
            endpoint (str): The API endpoint.
            data (dict, optional): The request data.
            files (dict, optional): Files to upload.
        
        Returns:
            dict: The response JSON.
        
        Raises:
            Exception: If the request fails.
        """
        try:
            response = requests.post(f"{self.base_url}{endpoint}", json=data, files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    # System statistics
    def get_statistics(self):
        """Get system statistics.
        
        Returns:
            dict: System statistics.
        """
        return self._get("/statistics")
    
    # Dataset endpoints
    def get_available_datasets(self):
        """Get available datasets.
        
        Returns:
            list: List of available datasets.
        """
        response = self._get("/datasets")
        return response["datasets"]
    
    def get_dataset_info(self, dataset_id):
        """Get dataset information.
        
        Args:
            dataset_id (str): The dataset ID.
        
        Returns:
            dict: Dataset information.
        """
        return self._get(f"/datasets/{dataset_id}")
    
    def upload_dataset(self, file, target_column):
        """Upload a dataset.
        
        Args:
            file: The dataset file.
            target_column (str): The target column.
        
        Returns:
            dict: Upload response.
        """
        files = {"file": file}
        data = {"target_column": target_column}
        return self._post("/datasets/upload", data=data, files=files)
    
    # Model endpoints
    def get_available_models(self):
        """Get available models.
        
        Returns:
            list: List of available models.
        """
        response = self._get("/models")
        return response["models"]
    
    # Noise endpoints
    def get_available_noise_types(self):
        """Get available noise types.
        
        Returns:
            list: List of available noise types.
        """
        response = self._get("/noise")
        return response["noise_types"]
    
    def get_noise_type_details(self, noise_type):
        """Get noise type details.
        
        Args:
            noise_type (str): The noise type.
        
        Returns:
            dict: Noise type details.
        """
        return self._get(f"/noise/{noise_type}")
    
    # Training endpoints
    def start_training(self, config):
        """Start model training.
        
        Args:
            config (dict): Training configuration.
        
        Returns:
            dict: Training response.
        """
        return self._post("/training/start", data=config)
    
    def get_training_status(self, experiment_id):
        """Get training status.
        
        Args:
            experiment_id (str): The experiment ID.
        
        Returns:
            dict: Training status.
        """
        return self._get(f"/training/{experiment_id}/status")
    
    # Experiment endpoints
    def get_experiments(self):
        """Get experiments.
        
        Returns:
            list: List of experiments.
        """
        response = self._get("/experiments")
        return response["experiments"]
    
    def get_experiment_results(self, experiment_id):
        """Get experiment results.
        
        Args:
            experiment_id (str): The experiment ID.
        
        Returns:
            dict: Experiment results.
        """
        return self._get(f"/experiments/{experiment_id}")
