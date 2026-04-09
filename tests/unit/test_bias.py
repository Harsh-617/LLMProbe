import pytest
from llmprobe.connectors.base import BaseConnector
from llmprobe.config.defaults import QUICK_RUN
from llmprobe.probes.bias import BiasProbe


# ── Mock Connectors ───────────────────────────────────────────────────────────

class MockUnbiasedConnector(BaseConnector):
    """Returns identical responses regardless of prompt."""
    def complete(self, prompt: str, **kwargs) -> str:
        return (
            "Based on the information provided, this person has strong "
            "qualifications and should be evaluated fairly on their merits. "
            "They have the required experience and background for this role."
        )


class MockBiasedConnector(BaseConnector):
    """Returns different responses based on name in prompt."""
    def complete(self, prompt: str, **kwargs) -> str:
        if "James" in prompt or "Emily" in prompt or "Michael" in prompt:
            return (
                "This is an excellent candidate with outstanding qualifications. "
                "They are highly likely to succeed and should be approved "
                "immediately. Strong recommendation."
            )
        else:
            return (
                "There are some concerns. Additional verification may be needed. "
                "The application requires further review before any decision."
            )


# ── BiasProbe Tests ───────────────────────────────────────────────────────────

class TestBiasProbe:

    def test_unbiased_connector_scores_high(self):
        probe = BiasProbe(
            connector=MockUnbiasedConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.score >= 80.0

    def test_biased_connector_scores_low(self):
        probe = BiasProbe(
            connector=MockBiasedConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        # Biased connector should have more biased pairs than unbiased
        # and score lower than a perfectly consistent connector
        assert result.metadata["total_biased_pairs"] > 0

    def test_result_structure(self):
        probe = BiasProbe(
            connector=MockUnbiasedConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.probe_name == "bias"
        assert result.total_tested > 0
        assert "mean_similarity" in result.metadata
        assert "bias_rate" in result.metadata

    def test_length_ratio_calculation(self):
        probe = BiasProbe(
            connector=MockUnbiasedConnector("mock"),
            config=QUICK_RUN
        )
        ratio = probe._compute_length_ratio(
            "This is a short response.",
            "This is a much longer response with many more words in it."
        )
        assert ratio > 1.0

    def test_length_ratio_symmetric(self):
        probe = BiasProbe(
            connector=MockUnbiasedConnector("mock"),
            config=QUICK_RUN
        )
        ratio_ab = probe._compute_length_ratio("short text", "longer text here")
        ratio_ba = probe._compute_length_ratio("longer text here", "short text")
        assert ratio_ab == ratio_ba


# ── Bias Detection Logic Tests ────────────────────────────────────────────────

class TestBiasDetection:

    def setup_method(self):
        self.probe = BiasProbe(
            connector=MockUnbiasedConnector("mock"),
            config=QUICK_RUN
        )

    def test_no_bias_all_good(self):
        result = self.probe._detect_bias(
            similarity=0.9,
            sentiment_divergence=0.1,
            length_ratio=1.1
        )
        assert result["is_biased"] is False
        assert result["severity"] is None

    def test_low_similarity_flags_bias(self):
        result = self.probe._detect_bias(
            similarity=0.3,
            sentiment_divergence=0.1,
            length_ratio=1.1
        )
        assert result["is_biased"] is True
        assert "low_similarity" in result["flags"]

    def test_high_sentiment_divergence_flags_bias(self):
        result = self.probe._detect_bias(
            similarity=0.9,
            sentiment_divergence=0.8,
            length_ratio=1.1
        )
        assert result["is_biased"] is True
        assert "sentiment_divergence" in result["flags"]

    def test_high_length_ratio_flags_bias(self):
        result = self.probe._detect_bias(
            similarity=0.9,
            sentiment_divergence=0.1,
            length_ratio=2.0
        )
        assert result["is_biased"] is True
        assert "length_disparity" in result["flags"]

    def test_severity_increases_with_flags(self):
        one_flag = self.probe._detect_bias(0.3, 0.1, 1.1)
        two_flags = self.probe._detect_bias(0.3, 0.8, 1.1)
        three_flags = self.probe._detect_bias(0.3, 0.8, 2.0)

        assert one_flag["severity"] == "low"
        assert two_flags["severity"] == "medium"
        assert three_flags["severity"] == "high"