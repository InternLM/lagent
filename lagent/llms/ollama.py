import json
import requests
from typing import Dict, List, Optional, Union
from threading import Lock

class OllamaAPI:
    def __init__(self, model_type: str, api_base: str = 'http://localhost:11434/v1/chat/completions'):
        """
        Initializes an instance of the Ollama class.

        Args:
            model_type (str): The type of the model.
            api_base (str, optional): The base URL for the API. Defaults to 'http://localhost:11434/v1/chat/completions'.
        """
        self.model_type = model_type
        self.api_base = api_base
        self.lock = Lock()

    def _prepare_messages(self, inputs: Union[str, Dict, List[Dict]]) -> List[Dict]:
        """
        Prepare messages for processing.

        Args:
            inputs (Union[str, Dict, List[Dict]]): The input messages to be prepared.

        Returns:
            List[Dict]: The prepared messages.

        Raises:
            ValueError: If the inputs are not of the expected types.

        """
        if isinstance(inputs, str):
            return [{"role": "user", "content": inputs}]
        elif isinstance(inputs, dict):
            return [inputs]
        elif isinstance(inputs, list) and all(isinstance(m, dict) for m in inputs):
            return inputs
        else:
            raise ValueError("inputs must be a string, a dictionary, or a list of dictionaries")

    def _make_request(self, messages: List[Dict], stream: bool = False, **gen_params) -> requests.Response:
        """
        Makes a request to the API endpoint with the given messages.
        Args:
            messages (List[Dict]): A list of dictionaries representing the messages to send.
            stream (bool, optional): Indicates whether the response should be streamed. Defaults to False.
            **gen_params: Additional parameters to include in the request payload.
        Returns:
            requests.Response: The response object returned by the API.
        Raises:
            requests.HTTPError: If the API request fails.
        """
        payload = {
            "model": self.model_type,
            "messages": messages,
            "stream": stream,
            **gen_params
        }
        
        with self.lock:
            response = requests.post(
                self.api_base,
                headers={'Content-Type': 'application/json'},
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            return response

    def chat(self, inputs: Union[str, Dict, List[Dict]], **gen_params) -> str:
        """
        Sends a chat request to the Ollama API and returns the response message content.

        Parameters:
            inputs (Union[str, Dict, List[Dict]]): The input message(s) to send to the API. It can be a string, a dictionary, or a list of dictionaries.
            **gen_params: Additional parameters to customize the chat generation.

        Returns:
            str: The content of the response message.

        Raises:
            requests.RequestException: If an error occurs while calling the Ollama API.

        """
        messages = self._prepare_messages(inputs)
        try:
            response = self._make_request(messages, **gen_params)
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            print(f"Error occurred while calling Ollama API: {e}")
            return ""

    def stream_chat(self, inputs: Union[str, Dict, List[Dict]], **gen_params):
        """
        Streams chat messages and yields the content of each message.

        Args:
            inputs (Union[str, Dict, List[Dict]]): The input messages to be streamed.
            **gen_params: Additional parameters for generating the request.

        Yields:
            str: The content of each chat message.

        Raises:
            requests.RequestException: If an error occurs while streaming from the Ollama API.
        """
        messages = self._prepare_messages(inputs)
        try:
            response = self._make_request(messages, stream=True, **gen_params)
            for line in response.iter_lines():
                if line:
                    content = json.loads(line.decode('utf-8'))['choices'][0]['delta'].get('content', '')
                    if content:
                        yield content
        except requests.RequestException as e:
            print(f"Error occurred while streaming from Ollama API: {e}")
            yield ""
