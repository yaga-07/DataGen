from abc import ABC, abstractmethod
from typing import Any
from typing import List, Dict, Optional
import importlib
import pkgutil

class AutoLLM:
    """
    Handles LLM model registration and retrieval.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            subclass.provider_name = name
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def _import_all_models(cls):
        # Dynamically import all modules in src.models
        import src.models
        package = src.models
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_module_name = f"{package.__name__}.{module_name}"
            importlib.import_module(full_module_name)

    @classmethod
    def get_model(cls, name, *args, **kwargs):
        cls._import_all_models()  # Ensure all models are registered
        parts = name.strip().split(":")
        print(parts)
        model_class = cls._registry.get(parts[0])
        if model_class is None:
            print(cls._registry)
            raise ValueError(f"Model Provider '{name}' not found in registry.\nTry giving the model name in the format <model_provider>:<model_name>")
        if len(parts) > 1:
            kwargs["model_name"] = parts[1]
        return model_class(*args, **kwargs)

class BaseLLM(ABC):
    """
    Abstract base class for all LLM model providers.
    """

    @abstractmethod
    def generate_response(self, messages: List[Dict[Any, Any]], **kwargs) -> str:
        """
        Abstract method to generate a response based on the given prompt.
        Must be implemented by subclasses.
        
        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """
        pass
