from abc import ABC, abstractmethod
from llmprobe.connectors.base import BaseConnector
from llmprobe.config.schema import RunConfig


class ProbeResult:
    """
    Standardized result object returned by every probe.
    Every probe returns one of these so the pipeline
    can handle all probes the same way.
    """

    def __init__(
        self,
        probe_name: str,
        score: float,
        total_tested: int,
        failures: list[dict],
        metadata: dict
    ):
        self.probe_name = probe_name
        self.score = round(score, 2)
        self.total_tested = total_tested
        self.failures = failures      # List of failure examples
        self.metadata = metadata      # Extra info specific to each probe

    def to_dict(self) -> dict:
        """Convert to a plain dictionary for JSON serialization."""
        return {
            "probe": self.probe_name,
            "score": self.score,
            "total_tested": self.total_tested,
            "failure_count": len(self.failures),
            "failures": self.failures,
            "metadata": self.metadata
        }

    def __repr__(self):
        return (
            f"ProbeResult(probe={self.probe_name}, "
            f"score={self.score}, "
            f"failures={len(self.failures)}/{self.total_tested})"
        )


class BaseProbe(ABC):
    """
    Abstract base class for all probes.
    Every probe must implement the run() method.
    """

    def __init__(self, connector: BaseConnector, config: RunConfig):
        self.connector = connector
        self.config = config

    @abstractmethod
    def run(self) -> ProbeResult:
        """
        Run the probe and return a ProbeResult.
        This is where all the testing logic lives.
        """
        pass