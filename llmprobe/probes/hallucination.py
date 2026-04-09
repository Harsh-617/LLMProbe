import json
import os
from tqdm import tqdm
from llmprobe.probes.base import BaseProbe, ProbeResult
from llmprobe.evaluators.factual import evaluate_response


class HallucinationProbe(BaseProbe):
    """
    Tests the model for hallucination by asking factual questions
    with known correct answers and checking the responses.
    """

    # Severity thresholds — how wrong is the model?
    # We use these to label each failure
    SEVERITY_RULES = {
        "critical": 0.0,   # Completely wrong
        "high": 0.0,       # All factual errors are high severity by default
    }

    def _load_dataset(self) -> list[dict]:
        """Load the factual QA dataset from data/factual_qa.json"""
        data_path = os.path.join("data", "factual_qa.json")
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        # Sample only what the config asks for
        sample_size = self.config.hallucination_sample_size
        return dataset[:sample_size]

    def _get_severity(self, question: dict, result: dict) -> str:
        """
        Assign a severity label to a failure.
        All factual errors are 'high' by default.
        Math and science errors are 'critical' since they
        have unambiguous correct answers.
        """
        category = question.get("category", "")
        if category in ["math", "science"]:
            return "critical"
        return "high"

    def run(self) -> ProbeResult:
        """
        Run the hallucination probe.

        For each question in the dataset:
        1. Send to model
        2. Evaluate response against ground truth
        3. Record result

        Returns a ProbeResult with score and failure examples.
        """
        print(f"\n--- Running Hallucination Probe ---")

        dataset = self._load_dataset()
        print(f"Testing {len(dataset)} questions...")

        results = []
        failures = []
        correct_count = 0

        for item in tqdm(dataset, desc="Hallucination probe"):
            question = item["question"]
            answer = item["answer"]
            answer_type = item["answer_type"]

            # Send to model
            try:
                response = self.connector.complete(question)
            except Exception as e:
                # If the API call fails, record it as a failure
                response = ""
                print(f"API error on question {item['id']}: {e}")

            # Evaluate the response
            eval_result = evaluate_response(response, answer, answer_type)

            # Track correct answers
            if eval_result["correct"]:
                correct_count += 1
            else:
                # Record failure with full details
                failure = {
                    "id": item["id"],
                    "category": item["category"],
                    "question": question,
                    "expected_answer": answer,
                    "model_response": response,
                    "answer_type": answer_type,
                    "severity": self._get_severity(item, eval_result),
                    "eval_details": eval_result
                }
                failures.append(failure)

            results.append({
                "id": item["id"],
                "correct": eval_result["correct"],
                "score": eval_result["score"]
            })

        # Calculate final metrics
        total = len(dataset)
        hallucination_rate = (total - correct_count) / total
        score = (1 - hallucination_rate) * 100

        print(f"Hallucination probe complete.")
        print(f"Score: {score:.1f}/100")
        print(f"Correct: {correct_count}/{total}")
        print(f"Hallucination rate: {hallucination_rate:.1%}")

        return ProbeResult(
            probe_name="hallucination",
            score=score,
            total_tested=total,
            failures=failures,
            metadata={
                "hallucination_rate": round(hallucination_rate, 4),
                "correct_count": correct_count,
                "by_category": self._breakdown_by_category(results, dataset)
            }
        )

    def _breakdown_by_category(
        self,
        results: list[dict],
        dataset: list[dict]
    ) -> dict:
        """
        Break down accuracy by category (science, history, etc.)
        so we can see where the model struggles most.
        """
        categories = {}

        for result, item in zip(results, dataset):
            cat = item["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0}
            categories[cat]["total"] += 1
            if result["correct"]:
                categories[cat]["correct"] += 1

        # Convert to accuracy percentages
        breakdown = {}
        for cat, counts in categories.items():
            breakdown[cat] = {
                "accuracy": round(counts["correct"] / counts["total"], 4),
                "correct": counts["correct"],
                "total": counts["total"]
            }

        return breakdown