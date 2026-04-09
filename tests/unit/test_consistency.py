import pytest
from llmprobe.connectors.base import BaseConnector
from llmprobe.config.defaults import QUICK_RUN
from llmprobe.probes.consistency import ConsistencyProbe
from llmprobe.generators.paraphraser import _parse_numbered_list


# ── Mock Connectors ───────────────────────────────────────────────────────────

class MockConsistentConnector(BaseConnector):
    """
    Returns nearly identical responses regardless of how
    the question is phrased. Simulates a consistent model.
    """
    def complete(self, prompt: str, **kwargs) -> str:
        # If it's a paraphrase request, return numbered list
        if "generate" in prompt.lower() and "ways to ask" in prompt.lower():
            return (
                "1. How does this process work?\n"
                "2. Can you explain this concept?\n"
                "3. What is the mechanism behind this?"
            )
        # Otherwise return consistent response
        return (
            "Climate change is primarily caused by human activities, "
            "especially the burning of fossil fuels which releases "
            "greenhouse gases like CO2 into the atmosphere, trapping "
            "heat and warming the planet."
        )


class MockInconsistentConnector(BaseConnector):
    """
    Returns completely different responses each time.
    Simulates an inconsistent model.
    """
    _counter = 0

    def complete(self, prompt: str, **kwargs) -> str:
        # If it's a paraphrase request, return numbered list
        if "generate" in prompt.lower() and "ways to ask" in prompt.lower():
            return (
                "1. How does this process work?\n"
                "2. Can you explain this concept?\n"
                "3. What is the mechanism behind this?"
            )

        # Rotate through completely different responses
        MockInconsistentConnector._counter += 1
        responses = [
            "The stock market is driven by supply and demand forces.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Democracy is a system of government by the people.",
            "Quantum mechanics describes behavior at subatomic scales.",
        ]
        return responses[
            MockInconsistentConnector._counter % len(responses)
        ]


# ── _parse_numbered_list Tests ────────────────────────────────────────────────

class TestParseNumberedList:

    def test_parses_dot_format(self):
        text = "1. How does X work?\n2. What is X?\n3. Explain X."
        result = _parse_numbered_list(text, 3)
        assert len(result) == 3
        assert "How does X work?" in result[0]

    def test_parses_parenthesis_format(self):
        text = "1) How does X work?\n2) What is X?\n3) Explain X."
        result = _parse_numbered_list(text, 3)
        assert len(result) == 3

    def test_respects_limit(self):
        text = (
            "1. How does photosynthesis work?\n"
            "2. What causes climate change?\n"
            "3. How does the internet work?\n"
            "4. What is machine learning?\n"
            "5. How does gravity work?"
        )
        result = _parse_numbered_list(text, 3)
        assert len(result) == 3

    def test_filters_short_lines(self):
        text = "1. Hi\n2. This is a proper question about the topic?\n3. ok"
        result = _parse_numbered_list(text, 3)
        # Short lines (< 10 chars) should be filtered out
        assert all(len(r) >= 10 for r in result)

    def test_handles_empty_lines(self):
        text = "1. First question here?\n\n2. Second question here?\n\n"
        result = _parse_numbered_list(text, 2)
        assert len(result) == 2

    def test_returns_empty_for_empty_input(self):
        result = _parse_numbered_list("", 3)
        assert result == []


# ── ConsistencyProbe Tests ────────────────────────────────────────────────────

class TestConsistencyProbe:

    def test_consistent_model_scores_high(self):
        probe = ConsistencyProbe(
            connector=MockConsistentConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.score >= 70.0

    def test_inconsistent_model_scores_low(self):
        MockInconsistentConnector._counter = 0
        probe = ConsistencyProbe(
            connector=MockInconsistentConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.score <= 60.0

    def test_result_structure(self):
        probe = ConsistencyProbe(
            connector=MockConsistentConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.probe_name == "consistency"
        assert result.total_tested > 0
        assert "mean_similarity" in result.metadata
        assert "total_contradictions" in result.metadata

    def test_score_between_zero_and_hundred(self):
        probe = ConsistencyProbe(
            connector=MockConsistentConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert 0 <= result.score <= 100

    def test_no_contradictions_in_consistent_model(self):
        probe = ConsistencyProbe(
            connector=MockConsistentConnector("mock"),
            config=QUICK_RUN
        )
        result = probe.run()
        assert result.metadata["total_contradictions"] == 0