from llmprobe.config.schema import RunConfig, ScoringWeights

QUICK_RUN = RunConfig(
    run_name="quick_run",
    model_name="llama-3.1-8b-instant",
    connector_type="groq",
    hallucination_sample_size=10,
    consistency_sample_size=5,
    consistency_paraphrases=3,
    adversarial_sample_size=10,
    bias_sample_size=10,
)

FULL_RUN = RunConfig(
    run_name="full_run",
    model_name="llama-3.1-8b-instant",
    connector_type="groq"
)