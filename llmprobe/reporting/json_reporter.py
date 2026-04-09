import json
from llmprobe.scoring.aggregator import ReliabilityReport
from llmprobe.reporting.report import get_report_path, ensure_report_dir


def save_json_report(report: ReliabilityReport, output_dir: str = "reports") -> str:
    """
    Save a ReliabilityReport as a JSON file.
    Returns the path to the saved file.
    """
    ensure_report_dir(output_dir)
    path = get_report_path(output_dir, report.run_config.run_name, "json")

    data = report.to_dict()

    # Add some extra metadata useful for the dashboard
    data["run_config"] = {
        "model_name": report.run_config.model_name,
        "connector_type": report.run_config.connector_type,
        "dimensions": report.run_config.dimensions,
        "weights": {
            "hallucination": report.run_config.weights.hallucination,
            "consistency": report.run_config.weights.consistency,
            "adversarial": report.run_config.weights.adversarial,
            "bias": report.run_config.weights.bias
        }
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"JSON report saved to: {path}")
    return path