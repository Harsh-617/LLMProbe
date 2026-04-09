import os
import anthropic
from dotenv import load_dotenv
from llmprobe.connectors.base import BaseConnector

load_dotenv()


class AnthropicConnector(BaseConnector):
    """
    Connector for Anthropic models (Claude Haiku, Sonnet, Opus etc.)
    Requires ANTHROPIC_API_KEY in your .env file.
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, **kwargs) -> str:
        """Send a prompt to Anthropic and return the response text."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text