import numpy as np
from typing import Dict, Any

class LabelNoise:
    """Class for adding label noise to the dataset."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the noise generator.
        
        Args:
            params (Dict[str, Any], optional): Noise parameters.
        """
        self.params = params or {}
        self.flip_ratio = self.params.get('flip_ratio', 0.1)
        self.random_state = self.params.get('random_state', 42)
    
    def apply(self, y: np.ndarray) -> np.ndarray:
        """Apply label noise to the target.
        
        Args:
            y (np.ndarray): Target.
        
        Returns:
            np.ndarray: Noisy target.
        """
        # Set random seed
        np.random.seed(self.random_state)
        
        # Make a copy of the target
        y_noisy = y.copy()
        
        # Get unique classes
        classes = np.unique(y)
        
        # Randomly select indices to flip
        n_samples = len(y)
        n_flip = int(n_samples * self.flip_ratio)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        
        # Flip labels
        for idx in flip_indices:
            # Get current class
            current_class = y[idx]
            
            # Get other classes
            other_classes = classes[classes != current_class]
            
            # Randomly select a new class
            new_class = np.random.choice(other_classes)
            
            # Flip the label
            y_noisy[idx] = new_class
        
        return y_noisy
