from typing import List, Dict, Any, Optional

from backend.core.config import settings
from ml.noise.noise_factory import NoiseFactory

class NoiseService:
    """Service for handling noise types."""
    
    def __init__(self):
        """Initialize the noise service."""
        self.available_noise_types = settings.AVAILABLE_NOISE_TYPES
    
    def get_available_noise_types(self) -> List[str]:
        """Get available noise types.
        
        Returns:
            List[str]: List of available noise types.
        """
        return [noise['name'] for noise in self.available_noise_types]
    
    def get_noise_type_details(self, noise_type: str) -> Optional[Dict[str, Any]]:
        """Get noise type details.
        
        Args:
            noise_type (str): The noise type.
        
        Returns:
            Optional[Dict[str, Any]]: Noise type details.
        """
        for noise in self.available_noise_types:
            if noise['name'] == noise_type:
                return noise
        
        return None
    
    def create_noise_generator(self, noise_type: str, noise_params: Dict[str, Any]) -> Optional[object]:
        """Create a noise generator.
        
        Args:
            noise_type (str): The noise type.
            noise_params (Dict[str, Any]): Noise parameters.
        
        Returns:
            Optional[object]: The created noise generator.
        """
        return NoiseFactory.create(noise_type, noise_params)
