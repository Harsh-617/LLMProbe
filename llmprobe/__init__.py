from llmprobe.pipeline.runner import PipelineRunner
from llmprobe.config.schema import RunConfig
from llmprobe.config.defaults import QUICK_RUN, FULL_RUN


class LLMProbe:
    """
    Main entry point for LLMProbe.

    Usage:
        from llmprobe import LLMProbe
        from llmprobe.config.schema import RunConfig

        probe = LLMProbe(config=RunConfig(
            model_name="llama-3.3-70b-versatile",
            connector_type="groq",
            dimensions=["hallucination", "consistency"]
        ))
        report = probe.run()
        print(report.summary())
    """

    def __init__(self, config: RunConfig = None):
        self.config = config or QUICK_RUN
        self.runner = PipelineRunner(self.config)

    def run(self):
        """Run the full evaluation pipeline and return a ReliabilityReport."""
        return self.runner.run()