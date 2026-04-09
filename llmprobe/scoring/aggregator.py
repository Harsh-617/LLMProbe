from llmprobe.probes.base import ProbeResult
from llmprobe.config.schema import RunConfig, ScoringWeights
from llmprobe.scoring.scorer import score_probe_result, get_rating, normalize_score


class ReliabilityReport:
    """
    The final output of a full LLMProbe run.

    Contains:
    - Overall reliability score
    - Per-dimension scores with ratings
    - All failure examples
    - Run metadata
    """

    def __init__(
        self,
        model_name: str,
        connector_type: str,
        overall_score: float,
        dimension_scores: dict,
        all_failures: list[dict],
        run_config: RunConfig
    ):
        self.model_name = model_name
        self.connector_type = connector_type
        self.overall_score = overall_score
        self.overall_rating = get_rating(overall_score)
        self.dimension_scores = dimension_scores
        self.all_failures = all_failures
        self.run_config = run_config

    def to_dict(self) -> dict:
        """Convert to plain dictionary for JSON serialization."""
        return {
            "model": self.model_name,
            "connector": self.connector_type,
            "overall_score": self.overall_score,
            "overall_rating": self.overall_rating["label"],
            "dimensions": {
                name: {
                    "score": data["score"],
                    "rating": data["rating"],
                    "failure_count": data["failure_count"],
                    "failure_rate": data["failure_rate"],
                    "metadata": data["metadata"]
                }
                for name, data in self.dimension_scores.items()
            },
            "total_failures": len(self.all_failures),
            "failures": self.all_failures
        }

    def summary(self) -> str:
        """
        Print a clean human readable summary of the report.
        Used after a run completes.
        """
        lines = [
            "",
            "=" * 50,
            f"  LLMProbe Reliability Report",
            f"  Model: {self.model_name}",
            "=" * 50,
            f"  Overall Score: {self.overall_score}/100 "
            f"[{self.overall_rating['label']}]",
            "",
            "  Dimension Scores:",
        ]

        for name, data in self.dimension_scores.items():
            bar = self._score_bar(data["score"])
            lines.append(
                f"    {name:<15} {data['score']:>6.1f}/100  "
                f"{bar}  [{data['rating']}]"
            )

        lines += [
            "",
            f"  Total failures found: {len(self.all_failures)}",
            "=" * 50,
        ]

        return "\n".join(lines)

    def _score_bar(self, score: float, width: int = 10) -> str:
        """Generate a simple ASCII progress bar for a score."""
        filled = int((score / 100) * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"


class ScoreAggregator:
    """
    Combines probe results into a final ReliabilityReport.
    Applies configurable weights to each dimension.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.weights = config.weights

    def aggregate(
        self,
        probe_results: list[ProbeResult]
    ) -> ReliabilityReport:
        """
        Take a list of ProbeResults and produce a ReliabilityReport.

        Steps:
        1. Score each probe result individually
        2. Apply weights to each dimension score
        3. Sum weighted scores for overall reliability score
        4. Collect all failures across dimensions
        5. Return structured ReliabilityReport
        """

        # Map probe names to their weights
        weight_map = {
            "hallucination": self.weights.hallucination,
            "consistency": self.weights.consistency,
            "adversarial": self.weights.adversarial,
            "bias": self.weights.bias
        }

        dimension_scores = {}
        all_failures = []
        weighted_total = 0.0
        total_weight_used = 0.0

        for result in probe_results:
            # Score the individual probe result
            scored = score_probe_result(result)
            dimension_scores[result.probe_name] = scored

            # Apply weight for this dimension
            weight = weight_map.get(result.probe_name, 0.0)
            weighted_total += scored["score"] * weight
            total_weight_used += weight

            # Collect failures with dimension label
            for failure in result.failures:
                failure_with_dim = {
                    "dimension": result.probe_name,
                    **failure
                }
                all_failures.append(failure_with_dim)

        # If all 4 dimensions ran, total_weight_used = 1.0
        # weighted_total is already the final score
        # If only some dimensions ran, normalize by weight used
        if total_weight_used > 0:
            # weighted_total already contains (score × weight) sums
            # Dividing by total_weight_used normalizes for partial runs
            overall_score = normalize_score(
                weighted_total / total_weight_used
            )
        else:
            overall_score = 0.0

        return ReliabilityReport(
            model_name=self.config.model_name,
            connector_type=self.config.connector_type,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            all_failures=all_failures,
            run_config=self.config
        )