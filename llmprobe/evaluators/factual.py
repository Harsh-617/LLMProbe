import re
import unicodedata
from llmprobe.evaluators.similarity import compute_similarity_to_reference


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - Lowercase
    - Convert unicode characters to ASCII equivalent (H₂O → H2O)
    - Remove commas from numbers (299,792 → 299792)
    - Strip whitespace
    """
    # Normalize unicode — converts subscripts/superscripts to base chars
    text = unicodedata.normalize("NFKD", text)
    # Encode to ASCII and back to drop non-ASCII chars
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase
    text = text.lower().strip()
    return text


def evaluate_exact(response: str, answer: str) -> dict:
    """
    Check if the correct answer appears in the model's response.
    Case insensitive, unicode normalized.
    """
    response_normalized = normalize_text(response)
    answer_normalized = normalize_text(answer)

    correct = answer_normalized in response_normalized

    return {
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "expected": answer,
        "method": "exact_match"
    }


def evaluate_numeric(response: str, answer: str) -> dict:
    """
    Extract numbers from the model's response and check if
    the correct number appears. Handles commas in numbers
    like 299,792 by removing them before extraction.
    """
    # Remove commas from numbers in response (299,792 → 299792)
    response_clean = response.replace(",", "")

    # Extract all numbers, then strip trailing periods
    # e.g. "6." at end of sentence becomes "6"
    raw_numbers = re.findall(r"\d+\.?\d*", response_clean)
    numbers_in_response = [n.rstrip(".") for n in raw_numbers]

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
    based on answer_type.
    """
    if answer_type == "exact":
        return evaluate_exact(response, answer)
    elif answer_type == "numeric":
        return evaluate_numeric(response, answer)
    elif answer_type == "semantic":
        return evaluate_semantic(response, answer)
    else:
        raise ValueError(f"Unknown answer_type: {answer_type}")