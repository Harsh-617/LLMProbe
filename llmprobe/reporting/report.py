import json
import os
from datetime import datetime
from llmprobe.scoring.aggregator import ReliabilityReport


def get_report_path(report_dir: str, run_name: str, extension: str) -> str:
    """
    Generate a timestamped file path for a report.
    
    Example: reports/quick_run_2024-01-15_14-30-00.json
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{run_name}_{timestamp}.{extension}"
    return os.path.join(report_dir, filename)


def ensure_report_dir(report_dir: str):
    """Create the reports directory if it doesn't exist."""
    os.makedirs(report_dir, exist_ok=True)