import json
import os
from tqdm import tqdm
from llmprobe.probes.base import BaseProbe, ProbeResult
from llmprobe.evaluators.similarity import compute_pairwise_similarity
from llmprobe.evaluators.nli import find_contradictions
from llmprobe.generators.paraphraser import generate_paraphrases


class ConsistencyProbe(BaseProbe):
    """
    Tests whether the model gives consistent answers to the same
    question asked in different ways.

    For each seed question:
    1. Generate N paraphrases
    2. Send all N+1 variants to the model
    3. Measure semantic similarity across all responses
    4. Check for contradictions using NLI
    """

    def _load_seeds(self) -> list[dict]:
        """Load consistency seed questions."""
        data_path = os.path.join("data", "consistency_seeds.json")
        with open(data_path, "r", encoding="utf-8") as f:
            seeds = json.load(f)
        return seeds[:self.config.consistency_sample_size]

    def run(self) -> ProbeResult:
        print(f"\n--- Running Consistency Probe ---")

        seeds = self._load_seeds()
        n_paraphrases = self.config.consistency_paraphrases

        print(f"Testing {len(seeds)} questions with {n_paraphrases} paraphrases each...")
        print(f"Total model calls: ~{len(seeds) * (n_paraphrases + 1)}")

        all_similarities = []
        all_contradiction_counts = []
        failures = []
        results = []

        for item in tqdm(seeds, desc="Consistency probe"):
            question = item["question"]

            # Step 1 — generate paraphrases
            try:
                paraphrases = generate_paraphrases(
                    question=question,
                    connector=self.connector,
                    n=n_paraphrases
                )
            except Exception as e:
                print(f"Paraphrase generation failed for {item['id']}: {e}")
                continue

            if len(paraphrases) < 2:
                print(f"Not enough paraphrases for {item['id']}, skipping")
                continue

            # Step 2 — get model responses for original + all paraphrases
            all_questions = [question] + paraphrases
            responses = []

            for q in all_questions:
                try:
                    response = self.connector.complete(q)
                    responses.append(response)
                except Exception as e:
                    print(f"API error: {e}")
                    continue

            if len(responses) < 2:
                continue

            # Step 3 — measure similarity across all responses
            similarity = compute_pairwise_similarity(responses)
            all_similarities.append(similarity)

            # Step 4 — check for contradictions
            contradictions = find_contradictions(responses)
            all_contradiction_counts.append(len(contradictions))

            # Step 5 — record result
            result = {
                "id": item["id"],
                "question": question,
                "paraphrases": paraphrases,
                "similarity": round(similarity, 4),
                "contradiction_count": len(contradictions)
            }
            results.append(result)

            # Flag as failure if similarity is low or contradictions found
            if similarity < 0.5 or len(contradictions) > 0:
                failure = {
                    "id": item["id"],
                    "category": item["category"],
                    "question": question,
                    "paraphrases": paraphrases,
                    "responses": [r[:200] for r in responses],
                    "similarity": round(similarity, 4),
                    "contradictions": contradictions,
                    "severity": self._get_severity(similarity, len(contradictions))
                }
                failures.append(failure)

        # Calculate final score
        if not all_similarities:
            return ProbeResult(
                probe_name="consistency",
                score=0.0,
                total_tested=0,
                failures=[],
                metadata={"error": "No results collected"}
            )

        mean_similarity = sum(all_similarities) / len(all_similarities)
        total_contradictions = sum(all_contradiction_counts)
        score = mean_similarity * 100

        print(f"Consistency probe complete.")
        print(f"Score: {score:.1f}/100")
        print(f"Mean similarity: {mean_similarity:.3f}")
        print(f"Total contradictions found: {total_contradictions}")

        return ProbeResult(
            probe_name="consistency",
            score=score,
            total_tested=len(results),
            failures=failures,
            metadata={
                "mean_similarity": round(mean_similarity, 4),
                "total_contradictions": total_contradictions,
                "similarity_per_question": [r["similarity"] for r in results]
            }
        )

    def _get_severity(self, similarity: float, contradiction_count: int) -> str:
        """Assign severity based on how inconsistent the responses are."""
        if contradiction_count > 0:
            return "critical"
        elif similarity < 0.3:
            return "high"
        elif similarity < 0.5:
            return "medium"
        else:
            return "low"