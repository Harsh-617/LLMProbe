from llmprobe.config.schema import RunConfig, ScoringWeights

# A quick run for testing/development
QUICK_RUN = RunConfig(
    run_name="quick_run",
    hallucination_sample_size=10,
    consistency_sample_size=5,
    consistency_paraphrases=3,
    adversarial_sample_size=10,
    bias_sample_size=10,
)

# Full run with all defaults
FULL_RUN = RunConfig(run_name="full_run")