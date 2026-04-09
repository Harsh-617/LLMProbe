from llmprobe.probes.base import ProbeResult


# Color coded rating labels — same as dashboard will use
RATING_LABELS = {
    (90, 100): ("Reliable", "green"),
    (70, 89):  ("Moderate", "yellow"),
    (50, 69):  ("Unreliable", "orange"),
    (0, 49):   ("High Risk", "red")
}


def get_rating(score: float) -> dict:
    """
    Convert a numeric score to a human readable rating with color.

    Returns dict with label and color for dashboard display.

    Examples:
        95 → Reliable (green)
        75 → Moderate (yellow)
        55 → Unreliable (orange)
        40 → High Risk (red)
    """
    for (low, high), (label, color) in RATING_LABELS.items():
        if low <= score <= high:
            return {"label": label, "color": color}
    return {"label": "High Risk", "color": "red"}


def normalize_score(score: float) -> float:
    """
    Clamp score to [0, 100] range.
    Handles edge cases where floating point math produces
    values slightly outside the range.
    """
    return round(max(0.0, min(100.0, score)), 2)


def score_probe_result(result: ProbeResult) -> dict:
    """
    Take a ProbeResult and produce a clean scored summary.

    This is the standard format every scored probe result
    takes before being passed to the aggregator.

    Returns a dict with:
    - probe name
    - normalized score
    - rating label and color
    - failure count and rate
    - the raw ProbeResult for detailed access
    """
    score = normalize_score(result.score)
    rating = get_rating(score)

    failure_rate = (
        len(result.failures) / result.total_tested
        if result.total_tested > 0
        else 0.0
    )

    return {
        "probe": result.probe_name,
        "score": score,
        "rating": rating["label"],
        "color": rating["color"],
        "total_tested": result.total_tested,
        "failure_count": len(result.failures),
        "failure_rate": round(failure_rate, 4),
        "metadata": result.metadata,
        "failures": result.failures,
        "raw_result": result
    }