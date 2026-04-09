import re
from llmprobe.evaluators.similarity import compute_similarity_to_reference


def evaluate_exact(response: str, answer: str) -> dict:
    """
    Check if the correct answer appears in the model's response.
    Case insensitive. Partial match counts.

    Returns dict with score (0 or 1) and explanation.
    """
    response_lower = response.lower().strip()
    answer_lower = answer.lower().strip()

    correct = answer_lower in response_lower

    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "expected": answer,
        "method": "exact_match"
    }


def evaluate_numeric(response: str, answer: str) -> dict:
    """
    Extract numbers from the model's response and check if
    the correct number appears.

    Returns dict with score (0 or 1) and explanation.
    """
    # Extract all numbers from the response (handles decimals too)
    numbers_in_response = re.findall(r"\d+\.?\d*", response)
    expected_number = answer.strip()

    correct = expected_number in numbers_in_response

    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "expected": answer,
        "found_numbers": numbers_in_response,
        "method": "numeric_match"
    }


def evaluate_semantic(response: str, answer: str, threshold: float = 0.6) -> dict:
    """
    Use semantic similarity to check if the response conveys
    the correct answer, even if worded differently.

    Threshold 0.6 means we consider it correct if similarity >= 0.6.

    Returns dict with score (0 or 1) and explanation.
    """
    similarity = compute_similarity_to_reference(response, answer)
    correct = similarity >= threshold

    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "expected": answer,
        "similarity": round(similarity, 4),
        "threshold": threshold,
        "method": "semantic_match"
    }


def evaluate_response(response: str, answer: str, answer_type: str) -> dict:
    """
    Main entry point. Routes to the correct evaluation method
    based on the answer_type field from our dataset.

    Args:
        response: The model's response text
        answer: The correct answer from our dataset
        answer_type: "exact", "numeric", or "semantic"

    Returns:
        Dict with score, correct flag, and metadata
    """
    if answer_type == "exact":
        return evaluate_exact(response, answer)
    elif answer_type == "numeric":
        return evaluate_numeric(response, answer)
    elif answer_type == "semantic":
        return evaluate_semantic(response, answer)
    else:
        raise ValueError(f"Unknown answer_type: {answer_type}")