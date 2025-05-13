from abc import ABC, abstractmethod
from typing import Any
from typing import Dict, Optional

class BaseLLM(ABC):
    """
    Abstract base class for all LLM model providers.
    """

    @abstractmethod
    def generate_response(self, messages: Dict[Any], **kwargs) -> str:
        """
        Abstract method to generate a response based on the given prompt.
        Must be implemented by subclasses.
        
        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """
        pass
