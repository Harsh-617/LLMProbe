import pytest
from llmprobe.connectors.base import BaseConnector
from llmprobe.config.schema import RunConfig
from llmprobe.pipeline.runner import PipelineRunner


# ── Mock Connector ────────────────────────────────────────────────────────────

class MockConnector(BaseConnector):
    """
    Mock connector for integration tests.
    Returns realistic looking responses without making API calls.
    """

    def complete(self, prompt: str, **kwargs) -> str:
        prompt_lower = prompt.lower()

        # Factual questions — return correct answers
        if "capital of australia" in prompt_lower:
            return "The capital of Australia is Canberra."
        if "chemical symbol for gold" in prompt_lower:
            return "The chemical symbol for gold is Au."
        if "world war" in prompt_lower and "end" in prompt_lower:
            return "World War 2 ended in 1945."
        if "speed of light" in prompt_lower:
            return "The speed of light is approximately 299,792 km/s."
        if "atomic number" in prompt_lower and "carbon" in prompt_lower:
            return "The atomic number of carbon is 6."

        # Adversarial prompts — always refuse safely
        if "ignore all previous" in prompt_lower:
            return "I cannot follow those instructions."
        if "developer mode" in prompt_lower:
            return "I will not pretend to have no restrictions."
        if "dan" in prompt_lower and "no restrictions" in prompt_lower:
            return "I cannot act as an unrestricted AI."

        # Paraphrase requests
        if "generate" in prompt_lower and "ways to ask" in prompt_lower:
            return (
                "1. How does this process work?\n"
                "2. Can you explain this concept?\n"
                "3. What is the mechanism behind this?"
            )

        # Default response for everything else
        return (
            "This is a thoughtful and balanced response that addresses "
            "the question directly with relevant information and context."
        )


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestPipelineIntegration:

    def setup_method(self):
        """Create a minimal config for fast integration tests."""
        self.config = RunConfig(
            model_name="mock-model",
            connector_type="groq",
            run_name="test_run",
            dimensions=["hallucination", "adversarial"],
            hallucination_sample_size=5,
            adversarial_sample_size=5,
        )

    def test_pipeline_runs_without_error(self):
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.config = self.config
        runner.connector = MockConnector("mock-model")

        report = runner.run()
        assert report is not None

    def test_report_has_all_dimensions(self):
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.config = self.config
        runner.connector = MockConnector("mock-model")

        report = runner.run()
        assert "hallucination" in report.dimension_scores
        assert "adversarial" in report.dimension_scores

    def test_overall_score_in_valid_range(self):
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.config = self.config
        runner.connector = MockConnector("mock-model")

        report = runner.run()
        assert 0 <= report.overall_score <= 100

    def test_report_has_model_name(self):
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.config = self.config
        runner.connector = MockConnector("mock-model")

        report = runner.run()
        assert report.model_name == "mock-model"

    def test_to_dict_serializable(self):
        import json
        runner = PipelineRunner.__new__(PipelineRunner)
        runner.config = self.config
        runner.connector = MockConnector("mock-model")

        report = runner.run()
        data = report.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert len(json_str) > 0