from ..core import BaseLLM
from typing import Optional, List, Dict, Any
import os

class GoogleLLM(BaseLLM):
    """
    A class to interact with Google LLMs (Vertex AI Gemini) using either service account or API key.
    """

    def __init__(self, model_name: str, api_key: Optional[str] = None, service_account_json: Optional[str] = None):
        """
        Initialize the GoogleLLM with a model name and either an API key or a service account JSON key.

        :param model_name: The name of the Google model.
        :param api_key: Optional API key for authentication.
        :param service_account_json: Optional path to service account JSON key.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.service_account_json = service_account_json or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if self.service_account_json:
            # Service account authentication
            from google.cloud import aiplatform
            from google.oauth2 import service_account
            from vertexai.generative_models import GenerativeModel
            project_id = os.environ.get("GOOGLE_PROJECT_ID", "")
            location = os.environ.get("GOOGLE_LOCATION", "")
            credentials = service_account.Credentials.from_service_account_file(self.service_account_json)
            aiplatform.init(project=project_id, location=location, credentials=credentials)
            self.client = GenerativeModel(model_name=self.model_name)
            self._mode = "service_account"
        elif self.api_key:
            # API key authentication
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self._mode = "api_key"
        else:
            raise ValueError("Either service_account_json or api_key must be provided.")

    def generate_response(self, messages: List[Dict[Any, Any]], **kwargs) -> str:
        """
        Generate a response from the model based on the given prompt.

        :param messages: The input prompt for the model.
        :param kwargs: Additional arguments for the model.
        :return: The generated response as a string.
        """
        if self._mode == "service_account":
            # For Vertex AI GenerativeModel
            # Assume messages is a list of dicts with 'content' keys
            prompt = "\n".join([msg.get("content", "") for msg in messages])
            response = self.client.generate_content(prompt, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        elif self._mode == "api_key":
            # For genai.Client
            prompt = "\n".join([msg.get("content", "") for msg in messages])
            response = self.client.generate_content(model=self.model_name, prompt=prompt, **kwargs)
            return response.text if hasattr(response, "text") else str(response)
        else:
            raise RuntimeError("Client not initialized properly.")

    def __repr__(self):
        return f"GoogleLLM(model_name={self.model_name}, mode={self._mode})"
