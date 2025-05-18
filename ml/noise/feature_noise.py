import numpy as np
from typing import Dict, Any

class GaussianNoise:
    """Class for adding Gaussian noise to features."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the noise generator.
        
        Args:
            params (Dict[str, Any], optional): Noise parameters.
        """
        self.params = params or {}
        self.mean = self.params.get('mean', 0.0)
        self.std = self.params.get('std', 0.1)
        self.random_state = self.params.get('random_state', 42)
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to features.
        
        Args:
            X (np.ndarray): Features.
        
        Returns:
            np.ndarray: Noisy features.
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Make a copy of the features
        X_noisy = X.copy()
        
        # Add Gaussian noise
        noise = np.random.normal(self.mean, self.std, X.shape)
        X_noisy += noise
        
        return X_noisy

class MissingValues:
    """Class for introducing missing values in the dataset."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the noise generator.
        
        Args:
            params (Dict[str, Any], optional): Noise parameters.
        """
        self.params = params or {}
        self.missing_ratio = self.params.get('missing_ratio', 0.1)
        self.random_state = self.params.get('random_state', 42)
    
    def apply(self, X: np.ndarray) -> np.ndarray:
        """Introduce missing values in the dataset.
        
        Args:
            X (np.ndarray): Features.
        
        Returns:
            np.ndarray: Features with missing values.
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Make a copy of the features
        X_noisy = X.copy()
        
        # Get total number of elements
        n_elements = X.size
        
        # Calculate number of elements to set as missing
        n_missing = int(n_elements * self.missing_ratio)
        
        # Randomly select indices to set as missing
        flat_indices = np.random.choice(n_elements, n_missing, replace=False)
        
        # Convert flat indices to multi-dimensional indices
        indices = np.unravel_index(flat_indices, X.shape)
        
        # Set selected elements as NaN
        X_noisy[indices] = np.nan
        
        return X_noisy
