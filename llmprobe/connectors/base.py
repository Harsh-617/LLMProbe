from abc import ABC, abstractmethod


class BaseConnector(ABC):
    """
    Abstract base class for all model connectors.
    Every connector (OpenAI, Anthropic, Groq, HuggingFace) must
    implement these methods. This guarantees the rest of LLMProbe
    can work with any connector interchangeably.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Send a single prompt to the model and return the response as a string.

        Args:
            prompt: The input text to send to the model
            **kwargs: Optional parameters like temperature, max_tokens etc.

        Returns:
            The model's response as a plain string
        """
        pass

    def batch_complete(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Send multiple prompts and return a list of responses.
        Default implementation just calls complete() in a loop.
        Connectors can override this for true parallel/batch support.

        Args:
            prompts: List of input prompts
            **kwargs: Passed through to complete()

        Returns:
            List of responses in the same order as prompts
        """
        return [self.complete(prompt, **kwargs) for prompt in prompts]

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"