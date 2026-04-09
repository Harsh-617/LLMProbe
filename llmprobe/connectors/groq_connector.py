import os
from groq import Groq
from dotenv import load_dotenv
from llmprobe.connectors.base import BaseConnector

load_dotenv()


class GroqConnector(BaseConnector):
    """
    Connector for Groq-hosted models (LLaMA, Mistral, Gemma etc.)
    Groq is free to use with rate limits — good for development.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def complete(self, prompt: str, **kwargs) -> str:
        """Send a prompt to Groq and return the response text."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return response.choices[0].message.content