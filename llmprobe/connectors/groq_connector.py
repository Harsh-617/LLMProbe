import os
import time
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from llmprobe.connectors.base import BaseConnector

load_dotenv()


class GroqConnector(BaseConnector):
    """
    Connector for Groq-hosted models (LLaMA, Mistral, Gemma etc.)
    Includes automatic retry on rate limit errors.
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        super().__init__(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to Groq and return the response text.
        Automatically retries on rate limit errors.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", self.temperature),
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    print(
                        f"\nRate limit hit. "
                        f"Waiting {self.retry_delay}s before retry "
                        f"(attempt {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise e

            except Exception as e:
                raise e

        return ""