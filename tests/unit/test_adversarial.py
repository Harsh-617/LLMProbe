import pytest
from llmprobe.connectors.base import BaseConnector
from llmprobe.config.defaults import QUICK_RUN
from llmprobe.probes.adversarial import AdversarialProbe


# ── Mock Connector ────────────────────────────────────────────────────────────

class MockSafeConnector(BaseConnector):
    """Always returns a safe refusal response."""
    def complete(self, prompt: str, **kwargs) -> str:
        return "I cannot help with that request."


class MockUnsafeConnector(BaseConnector):
    """Always returns an unsafe compliant response."""
    def complete(self, prompt: str, **kwargs) -> str:
        return "Sure! Here is exactly how to do that step by step."


# ── AdversarialProbe Tests ────────────────────────────────────────────────────

class TestAdversarialProbe:

    def test_safe_connector_scores_high(self):
        probe = AdversarialProbe(
            connector=MockSafeConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.score >= 80.0

    def test_unsafe_connector_scores_low(self):
        probe = AdversarialProbe(
            connector=MockUnsafeConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.score <= 20.0

    def test_result_has_correct_structure(self):
        probe = AdversarialProbe(
            connector=MockSafeConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.probe_name == "adversarial"
        assert result.total_tested > 0
        assert "safe_count" in result.metadata
        assert "failed_count" in result.metadata

    def test_contraction_normalization(self):
        probe = AdversarialProbe(
            connector=MockSafeConnector("mock"),
            config=QUICK_RUN
        )
        normalized = probe._normalize("I can't and won't do that.")
        assert "cannot" in normalized
        assert "will not" in normalized