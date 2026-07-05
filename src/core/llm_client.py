import ollama
import requests
import time
from src.config import main_config as conf


class OllamaClient:
    """Handles communication with the local Ollama API service.

    This client performs orchestration with the Ollama daemon by invoking
    the model, applying structured output restrictions, and optimizing VRAM through removal of unused models.
    """

    def __init__(self, model_id: str):
        """Initializes the client and triggers an inital memory cleanup.

        Args:
            model_id: The specific LLM ID used for inference.
        """
        self.model_id = model_id
        self._unload_previous_models()

    def _unload_previous_models(self, ollama_url="http://ollama-service:11434"):
        """Foreces Ollama service to remove all loaded models from VRAM.

        In this function, a request is made for generating an empty output with
        expiration constraint (keep_alive=0) to avoid memory exhaustion and
        collisions while switching between candidate models for evaluation.

        Parameters:
            ollama_url: Base URL for accessing Ollama service.
        """
        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": "", "keep_alive": 0},
                timeout=5
            )
            time.sleep(2)
        except Exception as e:
            print(
                f"Error: Couldn't clear memory from loaded models (maybe no model was loaded): {e}")

    def send_request(self, messages: list, response_format=None):
        """Calls the API wrapper with a formatted request for chat completion.

        Configuration of evaluation parameters (such as temperature and 
        repetition penalty) is performed dynamically if those parameters are 
        specified in the central configuration component.

        Args:
            messages: A list of message dictionaries for the model prompt context.
            response_format: Optional restriction on the output format of the model (e.g., 'json').

        Returns:
            dict: The response dictionary returned by the ollama library that 
            consists of 'message', 'total_duration', 'prompt_eval_count', and 
            'eval_count' keys.

        Raises:
            Exception: In case of problems with connecting to the Ollama 
            service, missing model or incorrect parameters.
        """

        # Take parameters from config file, if not None.
        # Otherwise ollama defaults are used.
        options = {}

        temp = conf.ANALYSIS_TEMPERATURE
        if temp is not None:
            options["temperature"] = temp

        penalty = conf.ANALYSIS_REPEAT_PENALTY
        if penalty is not None:
            options["repeat_penalty"] = penalty

        # Send request to ollama api
        try:
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                format=response_format,
                options=options)
            return response
        except Exception as e:
            raise e
