from src.core import BaseLLM, AutoModel
from typing import Optional, List, Dict, Any
from huggingface_hub import InferenceClient
import os

@AutoModel.register("hf")
class HFLLM(BaseLLM):
    """
    A class to interact with Hugging Face models using the Inference API.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the HFLLM with a model name and optional API key.

        :param model_name: The name of the Hugging Face model.
        :param api_key: Optional API key for authentication.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.client = InferenceClient(model=model_name, token=self.api_key)

    def generate_response(self, messages: List[Dict[Any, Any]], **kwargs) -> str:
        """
        Generate a response from the model based on the given prompt.

        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """

        completion = self.client.chat_completion(
            messages = messages,
            model = self.model_name, 
            **kwargs
        )
        
        return completion.choices[0].message.content
    
    def __repr__(self):
        return f"HFLLM(model_name={self.model_name})"