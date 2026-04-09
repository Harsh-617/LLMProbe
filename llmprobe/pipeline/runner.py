from llmprobe.config.schema import RunConfig
from llmprobe.connectors.base import BaseConnector
from llmprobe.connectors.groq_connector import GroqConnector
from llmprobe.connectors.openai_connector import OpenAIConnector
from llmprobe.connectors.anthropic_connector import AnthropicConnector
from llmprobe.probes.hallucination import HallucinationProbe
from llmprobe.probes.consistency import ConsistencyProbe
from llmprobe.probes.adversarial import AdversarialProbe
from llmprobe.probes.bias import BiasProbe
from llmprobe.scoring.aggregator import ScoreAggregator, ReliabilityReport
from llmprobe.reporting.json_reporter import save_json_report
from llmprobe.reporting.markdown_reporter import save_markdown_report


# Map dimension names to probe classes
PROBE_MAP = {
    "hallucination": HallucinationProbe,
    "consistency": ConsistencyProbe,
    "adversarial": AdversarialProbe,
    "bias": BiasProbe
}

# Map connector type strings to connector classes
CONNECTOR_MAP = {
    "groq": GroqConnector,
    "openai": OpenAIConnector,
    "anthropic": AnthropicConnector
}


def build_connector(config: RunConfig) -> BaseConnector:
    """
    Build the appropriate connector based on config.
    
    Looks up the connector class from CONNECTOR_MAP
    and instantiates it with the model name from config.
    """
    connector_class = CONNECTOR_MAP.get(config.connector_type)
    if connector_class is None:
        raise ValueError(
            f"Unknown connector type: {config.connector_type}. "
            f"Valid options: {list(CONNECTOR_MAP.keys())}"
        )
    return connector_class(model_name=config.model_name)


class PipelineRunner:
    """
    Orchestrates a full LLMProbe evaluation run.

    Usage:
        runner = PipelineRunner(config)
        report = runner.run()
        print(report.summary())
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.connector = build_connector(config)

    def run(self) -> ReliabilityReport:
        """
        Run all configured probe dimensions and return
        a ReliabilityReport with scores and failures.

        Steps:
        1. Print run header
        2. Run each probe in order
        3. Aggregate scores
        4. Return report
        """
        self._print_header()

        probe_results = []

        for dimension in self.config.dimensions:
            probe_class = PROBE_MAP.get(dimension)
            if probe_class is None:
                print(f"Warning: unknown dimension '{dimension}', skipping")
                continue

            try:
                probe = probe_class(
                    connector=self.connector,
                    config=self.config
                )
                result = probe.run()
                probe_results.append(result)

            except Exception as e:
                print(f"Error running {dimension} probe: {e}")
                print(f"Skipping {dimension} and continuing...")
                continue

        if not probe_results:
            raise RuntimeError(
                "No probes completed successfully. "
                "Check your API key and config."
            )

        # Aggregate all probe results into final report
        aggregator = ScoreAggregator(config=self.config)
        report = aggregator.aggregate(probe_results)

        # Save reports to disk
        save_json_report(report, output_dir=self.config.output_dir)
        save_markdown_report(report, output_dir=self.config.output_dir)

        return report

        return report

    def _print_header(self):
        """Print a clean header when a run starts."""
        print()
        print("=" * 50)
        print("  LLMProbe Evaluation Run")
        print(f"  Model:      {self.config.model_name}")
        print(f"  Connector:  {self.config.connector_type}")
        print(f"  Dimensions: {', '.join(self.config.dimensions)}")
        print(f"  Run name:   {self.config.run_name}")
        print("=" * 50)