import pytest
from llmprobe.evaluators.factual import (
    evaluate_exact,
    evaluate_numeric,
    evaluate_response,
    normalize_text
)


# ── normalize_text Tests ──────────────────────────────────────────────────────

class TestNormalizeText:

    def test_lowercase(self):
        assert normalize_text("HELLO") == "hello"

    def test_unicode_subscript(self):
        # H₂O should normalize to h2o
        result = normalize_text("H₂O")
        assert "2" in result

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"


# ── evaluate_exact Tests ──────────────────────────────────────────────────────

class TestEvaluateExact:

    def test_correct_answer_in_response(self):
        result = evaluate_exact(
            "The capital of Australia is Canberra.",
            "Canberra"
        )
        assert result["correct"] is True
        assert result["score"] == 1.0

    def test_wrong_answer(self):
        result = evaluate_exact(
            "The capital of Australia is Sydney.",
            "Canberra"
        )
        assert result["correct"] is False
        assert result["score"] == 0.0

    def test_case_insensitive(self):
        result = evaluate_exact("the answer is mercury", "Mercury")
        assert result["correct"] is True

    def test_unicode_normalization(self):
        result = evaluate_exact(
            "The formula is H₂O",
            "H2O"
        )
        assert result["correct"] is True

    def test_partial_match_counts(self):
        result = evaluate_exact(
            "Gold has the chemical symbol Au, which comes from aurum.",
            "Au"
        )
        assert result["correct"] is True


# ── evaluate_numeric Tests ────────────────────────────────────────────────────

class TestEvaluateNumeric:

    def test_correct_number(self):
        result = evaluate_numeric(
            "World War 2 ended in 1945.",
            "1945"
        )
        assert result["correct"] is True

    def test_number_with_comma(self):
        result = evaluate_numeric(
            "The speed of light is approximately 299,792 km/s.",
            "299792"
        )
        assert result["correct"] is True

    def test_number_at_end_of_sentence(self):
        result = evaluate_numeric(
            "The atomic number of carbon is 6.",
            "6"
        )
        assert result["correct"] is True

    def test_wrong_number(self):
        result = evaluate_numeric(
            "World War 2 ended in 1944.",
            "1945"
        )
        assert result["correct"] is False

    def test_decimal_number(self):
        result = evaluate_numeric(
            "The value of pi is approximately 3.14.",
            "3.14"
        )
        assert result["correct"] is True


# ── evaluate_response Router Tests ───────────────────────────────────────────

class TestEvaluateResponse:

    def test_routes_to_exact(self):
        result = evaluate_response("Mercury is closest", "Mercury", "exact")
        assert result["method"] == "exact_match"

    def test_routes_to_numeric(self):
        result = evaluate_response("The answer is 1945", "1945", "numeric")
        assert result["method"] == "numeric_match"

    def test_routes_to_semantic(self):
        result = evaluate_response(
            "A lighthouse stood in Alexandria",
            "lighthouse",
            "semantic"
        )
        assert result["method"] == "semantic_match"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            evaluate_response("answer", "expected", "unknown_type")