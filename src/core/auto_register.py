import importlib
import pkgutil
from typing import Any, Callable, Dict, Optional, Type

class AutoModel:
    """
    Handles registration and retrieval of model provider classes.
    Dynamically imports all modules in the `src.models` package to ensure all models are registered.
    """
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        """
        Decorator to register a model provider class with a given name.

        Args:
            name (str): The provider name to register the class under.

        Returns:
            Callable[[Type], Type]: The class decorator.
        """
        def decorator(subclass: Type) -> Type:
            subclass.provider_name = name
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def _import_all_models(cls) -> None:
        """
        Dynamically imports all modules in the `src.models` package to ensure all model providers are registered.
        """
        import src.models
        package = src.models
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_module_name = f"{package.__name__}.{module_name}"
            importlib.import_module(full_module_name)

    @classmethod
    def get_model(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieves an instance of a registered model provider class by name.

        Args:
            name (str): The model provider name, optionally in the format '<provider>:<model_name>'.
            *args: Positional arguments to pass to the model class constructor.
            **kwargs: Keyword arguments to pass to the model class constructor.

        Returns:
            Any: An instance of the requested model provider class.

        Raises:
            ValueError: If the model provider is not found in the registry.
        """
        cls._import_all_models()  # Ensure all models are registered
        parts = name.strip().split(":")
        model_class = cls._registry.get(parts[0])
        if model_class is None:
            print(cls._registry)
            raise ValueError(f"Model Provider '{name}' not found in registry.\nTry giving the model name in the format <model_provider>:<model_name>")
        if len(parts) > 1:
            kwargs["model_name"] = parts[1]
        return model_class(*args, **kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        """
        Returns a list of all registered model provider names.

        Returns:
            list[str]: List of registered model provider names.
        """
        cls._import_all_models()
        return list(cls._registry.keys())

class AutoTask:
    """
    Handles registration and retrieval of task classes.
    Dynamically imports all modules in the `src.tasks` package to ensure all tasks are registered.
    """
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        """
        Decorator to register a task class with a given name.

        Args:
            name (str): The task name to register the class under.

        Returns:
            Callable[[Type], Type]: The class decorator.
        """
        def decorator(subclass: Type) -> Type:
            subclass.task_name = name
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def _import_all_tasks(cls) -> None:
        """
        Dynamically imports all modules in the `src.tasks` package to ensure all tasks are registered.
        """
        import src.tasks
        package = src.tasks
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_module_name = f"{package.__name__}.{module_name}"
            importlib.import_module(full_module_name)

    @classmethod
    def get_task(cls, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Retrieves an instance of a registered task class by name.

        Args:
            name (str): The task name to retrieve.
            *args: Positional arguments to pass to the task class constructor.
            **kwargs: Keyword arguments to pass to the task class constructor.

        Returns:
            Any: An instance of the requested task class.

        Raises:
            ValueError: If the task is not found in the registry.
        """
        cls._import_all_tasks()  # Ensure all tasks are registered
        task_class = cls._registry.get(name)
        if task_class is None:
            raise ValueError(f"Task '{name}' not found in registry.")
        return task_class(*args, **kwargs)

    @classmethod
    def available_tasks(cls) -> list[str]:
        """
        Returns a list of all registered task names.

        Returns:
            list[str]: List of registered task names.
        """
        cls._import_all_tasks()
        return list(cls._registry.keys())

