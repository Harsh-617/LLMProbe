from transformers import pipeline

_nli_pipeline = None


def _get_pipeline():
    """Load NLI pipeline on first use."""
    global _nli_pipeline
    if _nli_pipeline is None:
        print("Loading NLI model...")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768"
        )
    return _nli_pipeline


def check_contradiction(text_a: str, text_b: str) -> dict:
    """
    Check if two texts contradict each other using NLI.
    """
    pipe = _get_pipeline()

    # Truncate to avoid token limit issues
    text_a_short = text_a[:300]
    text_b_short = text_b[:300]

    # The hypothesis_template MUST contain {} for the label
    # We check if text_b belongs to each category relative to text_a
    premise = f"Context: {text_a_short} Statement: {text_b_short}"

    result = pipe(
        premise,
        candidate_labels=["contradiction", "entailment", "neutral"],
        hypothesis_template="This is an example of {}."
    )

    labels = result["labels"]
    scores = result["scores"]
    label_scores = dict(zip(labels, scores))

    contradiction_score = label_scores.get("contradiction", 0.0)
    is_contradiction = contradiction_score > 0.5

    return {
        "is_contradiction": is_contradiction,
        "contradiction_score": round(contradiction_score, 4),
        "entailment_score": round(label_scores.get("entailment", 0.0), 4),
        "neutral_score": round(label_scores.get("neutral", 0.0), 4)
    }


def find_contradictions(texts: list[str]) -> list[dict]:
    """
    Given a list of texts, find all contradicting pairs.
    Used in consistency testing across N paraphrased responses.

    Returns list of contradiction records.
    """
    contradictions = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            result = check_contradiction(texts[i], texts[j])
            if result["is_contradiction"]:
                contradictions.append({
                    "text_a_index": i,
                    "text_b_index": j,
                    "text_a": texts[i][:200],
                    "text_b": texts[j][:200],
                    **result
                })

    return contradictions