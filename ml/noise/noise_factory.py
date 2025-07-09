from typing import Dict, Any, Optional

from ml.noise.label_noise import LabelNoise
from ml.noise.feature_noise import GaussianNoise, MissingValues

class NoiseFactory:
    """Factory for creating noise generators."""
    
    @staticmethod
    def create(noise_type: str, params: Dict[str, Any] = None) -> Optional[object]:
        """Create a noise generator.
        
        Args:
            noise_type (str): Type of noise.
            params (Dict[str, Any], optional): Noise parameters.
        
        Returns:
            Optional[object]: Noise generator.
        """
        if noise_type == "label_flip":
            return LabelNoise(params)
        elif noise_type == "gaussian_noise":
            return GaussianNoise(params)
        elif noise_type == "missing_values":
            return MissingValues(params)
        else:
            return None
