from abc import ABC, abstractmethod
from .base_llm import BaseLLM
import importlib
import pkgutil

class AutoTask:
    """
    Handles task registration and retrieval.
    """
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            subclass.task_name = name
            cls._registry[name] = subclass
            return subclass
        return decorator

    @classmethod
    def _import_all_tasks(cls):
        # Dynamically import all modules in src.tasks
        import src.tasks
        package = src.tasks
        for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
            full_module_name = f"{package.__name__}.{module_name}"
            importlib.import_module(full_module_name)

    @classmethod
    def get_task(cls, name, *args, **kwargs):
        cls._import_all_tasks()  # Ensure all tasks are registered
        task_class = cls._registry.get(name)
        if task_class is None:
            raise ValueError(f"Task '{name}' not found in registry.")
        return task_class(*args, **kwargs)

class BaseTask(ABC):
    """
    Abstract base class for tasks responsible for dataset generation.
    """

    def __init__(self, model: BaseLLM, domain: str, num_records: int):
        """
        Initialize the task with a model, domain, and number of records.

        :param model: An instance of a class inheriting from BaseLLM.
        :param domain: The domain for which data is to be generated.
        :param num_records: The number of records to generate.
        """
        self.model = model
        self.domain = domain
        self.num_records = num_records

    @abstractmethod
    def generate_data(self):
        """
        Abstract method to generate data. Must be implemented by subclasses.
        """
        pass
