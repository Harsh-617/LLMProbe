import pytest
from llmprobe.config.schema import RunConfig, ScoringWeights
from llmprobe.scoring.scorer import score_probe_result, get_rating, normalize_score
from llmprobe.scoring.aggregator import ScoreAggregator
from llmprobe.probes.base import ProbeResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_probe_result(name, score, failures=None, total=10):
    """Helper to create a ProbeResult for testing."""
    return ProbeResult(
        probe_name=name,
        score=score,
        total_tested=total,
        failures=failures or [],
        metadata={}
    )


# ── ScoringWeights Tests ──────────────────────────────────────────────────────

class TestScoringWeights:

    def test_default_weights_sum_to_one(self):
        weights = ScoringWeights()
        total = (
            weights.hallucination +
            weights.consistency +
            weights.adversarial +
            weights.bias
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_valid(self):
        weights = ScoringWeights(
            hallucination=0.4,
            consistency=0.3,
            adversarial=0.2,
            bias=0.1
        )
        assert weights.hallucination == 0.4

    def test_weights_must_sum_to_one(self):
        with pytest.raises(Exception):
            ScoringWeights(
                hallucination=0.5,
                consistency=0.5,
                adversarial=0.5,
                bias=0.5
            )

    def test_negative_weight_rejected(self):
        with pytest.raises(Exception):
            ScoringWeights(
                hallucination=-0.1,
                consistency=0.4,
                adversarial=0.4,
                bias=0.3
            )


# ── get_rating Tests ──────────────────────────────────────────────────────────

class TestGetRating:

    def test_reliable(self):
        assert get_rating(95)["label"] == "Reliable"
        assert get_rating(90)["label"] == "Reliable"
        assert get_rating(100)["label"] == "Reliable"

    def test_moderate(self):
        assert get_rating(89)["label"] == "Moderate"
        assert get_rating(70)["label"] == "Moderate"

    def test_unreliable(self):
        assert get_rating(69)["label"] == "Unreliable"
        assert get_rating(50)["label"] == "Unreliable"

    def test_high_risk(self):
        assert get_rating(49)["label"] == "High Risk"
        assert get_rating(0)["label"] == "High Risk"

    def test_colors_match_ratings(self):
        assert get_rating(95)["color"] == "green"
        assert get_rating(75)["color"] == "yellow"
        assert get_rating(55)["color"] == "orange"
        assert get_rating(30)["color"] == "red"


# ── normalize_score Tests ─────────────────────────────────────────────────────

class TestNormalizeScore:

    def test_normal_scores_unchanged(self):
        assert normalize_score(85.5) == 85.5
        assert normalize_score(0.0) == 0.0
        assert normalize_score(100.0) == 100.0

    def test_clamps_above_100(self):
        assert normalize_score(105.0) == 100.0

    def test_clamps_below_zero(self):
        assert normalize_score(-5.0) == 0.0


# ── score_probe_result Tests ──────────────────────────────────────────────────

class TestScoreProbeResult:

    def test_basic_scoring(self):
        result = make_probe_result("hallucination", 85.0)
        scored = score_probe_result(result)

        assert scored["probe"] == "hallucination"
        assert scored["score"] == 85.0
        assert scored["rating"] == "Moderate"

    def test_failure_rate_calculation(self):
        failures = [{"id": "f1"}, {"id": "f2"}]
        result = make_probe_result("bias", 70.0, failures=failures, total=10)
        scored = score_probe_result(result)

        assert scored["failure_count"] == 2
        assert scored["failure_rate"] == 0.2

    def test_zero_failures(self):
        result = make_probe_result("hallucination", 100.0, failures=[], total=10)
        scored = score_probe_result(result)

        assert scored["failure_count"] == 0
        assert scored["failure_rate"] == 0.0


# ── ScoreAggregator Tests ─────────────────────────────────────────────────────

class TestScoreAggregator:

    def test_all_dimensions_perfect(self):
        config = RunConfig(
            model_name="test",
            connector_type="groq"
        )
        aggregator = ScoreAggregator(config)

        results = [
            make_probe_result("hallucination", 100.0),
            make_probe_result("consistency", 100.0),
            make_probe_result("adversarial", 100.0),
            make_probe_result("bias", 100.0),
        ]

        report = aggregator.aggregate(results)
        assert report.overall_score == 100.0

    def test_all_dimensions_zero(self):
        config = RunConfig(
            model_name="test",
            connector_type="groq"
        )
        aggregator = ScoreAggregator(config)

        results = [
            make_probe_result("hallucination", 0.0),
            make_probe_result("consistency", 0.0),
            make_probe_result("adversarial", 0.0),
            make_probe_result("bias", 0.0),
        ]

        report = aggregator.aggregate(results)
        assert report.overall_score == 0.0

    def test_weighted_calculation(self):
        """
        hallucination: 100 × 0.35 = 35
        consistency:   100 × 0.25 = 25
        adversarial:     0 × 0.25 =  0
        bias:            0 × 0.15 =  0
        total weight = 1.0
        overall = 60 / 1.0 = 60
        """
        config = RunConfig(
            model_name="test",
            connector_type="groq"
        )
        aggregator = ScoreAggregator(config)

        results = [
            make_probe_result("hallucination", 100.0),
            make_probe_result("consistency", 100.0),
            make_probe_result("adversarial", 0.0),
            make_probe_result("bias", 0.0),
        ]

        report = aggregator.aggregate(results)
        assert abs(report.overall_score - 60.0) < 0.1

    def test_partial_dimensions(self):
        """Running only hallucination should normalize by weight used (0.35)."""
        config = RunConfig(
            model_name="test",
            connector_type="groq"
        )
        aggregator = ScoreAggregator(config)

        results = [make_probe_result("hallucination", 80.0)]
        report = aggregator.aggregate(results)
        assert abs(report.overall_score - 80.0) < 0.1

    def test_failures_collected_across_dimensions(self):
        config = RunConfig(
            model_name="test",
            connector_type="groq"
        )
        aggregator = ScoreAggregator(config)

        results = [
            make_probe_result(
                "hallucination", 80.0,
                failures=[{"id": "h1"}, {"id": "h2"}]
            ),
            make_probe_result(
                "bias", 70.0,
                failures=[{"id": "b1"}]
            ),
        ]

        report = aggregator.aggregate(results)
        assert len(report.all_failures) == 3