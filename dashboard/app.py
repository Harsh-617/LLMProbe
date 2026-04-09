import gradio as gr
import json
import os
import glob
from datetime import datetime

from llmprobe import LLMProbe
from llmprobe.config.schema import RunConfig, ScoringWeights


# ─── Helpers ────────────────────────────────────────────────────────────────

def get_color(score: float) -> str:
    if score >= 90: return "🟢"
    if score >= 70: return "🟡"
    if score >= 50: return "🟠"
    return "🔴"


def get_rating(score: float) -> str:
    if score >= 90: return "Reliable"
    if score >= 70: return "Moderate"
    if score >= 50: return "Unreliable"
    return "High Risk"


def score_bar(score: float, width: int = 20) -> str:
    filled = int((score / 100) * width)
    return "█" * filled + "░" * (width - filled)


def list_past_reports() -> list[str]:
    """List all saved JSON reports."""
    reports = glob.glob("reports/*.json")
    reports.sort(reverse=True)
    return reports


# ─── Run Tab ────────────────────────────────────────────────────────────────

def run_evaluation(
    model_name,
    connector_type,
    run_hallucination,
    run_consistency,
    run_adversarial,
    run_bias,
    hallucination_samples,
    consistency_seeds,
    consistency_paraphrases,
    adversarial_samples,
    bias_samples,
    progress=gr.Progress()
):
    """Main function called when user clicks Run."""

    dimensions = []
    if run_hallucination: dimensions.append("hallucination")
    if run_consistency:   dimensions.append("consistency")
    if run_adversarial:   dimensions.append("adversarial")
    if run_bias:          dimensions.append("bias")

    if not dimensions:
        return (
            "⚠️ Please select at least one dimension to evaluate.",
            None, None, None
        )

    try:
        config = RunConfig(
            model_name=model_name,
            connector_type=connector_type,
            dimensions=dimensions,
            run_name=f"dashboard_run",
            hallucination_sample_size=int(hallucination_samples),
            consistency_sample_size=int(consistency_seeds),
            consistency_paraphrases=int(consistency_paraphrases),
            adversarial_sample_size=int(adversarial_samples),
            bias_sample_size=int(bias_samples),
        )
    except Exception as e:
        return f"⚠️ Config error: {e}", None, None, None

    progress(0, desc="Starting evaluation...")

    try:
        probe = LLMProbe(config=config)
        progress(0.1, desc="Running probes...")
        report = probe.run()
        progress(1.0, desc="Done!")
    except Exception as e:
        return f"❌ Run failed: {e}", None, None, None

    # Build scorecard output
    scorecard = _build_scorecard(report)

    # Build failures output
    failures_df = _build_failures_table(report)

    # Status message
    status = (
        f"✅ Run complete — "
        f"{get_color(report.overall_score)} "
        f"{report.overall_score}/100 {get_rating(report.overall_score)}"
    )

    return status, scorecard, failures_df, report.to_dict()


def _build_scorecard(report) -> str:
    """Build a markdown scorecard from a report."""
    lines = [
        f"## {get_color(report.overall_score)} Overall: "
        f"{report.overall_score}/100 — "
        f"{get_rating(report.overall_score)}",
        f"",
        f"`{score_bar(report.overall_score)}` {report.overall_score}%",
        f"",
        f"---",
        f"",
        f"### Dimension Breakdown",
        f"",
    ]

    for name, data in report.dimension_scores.items():
        s = data["score"]
        lines += [
            f"**{name.capitalize()}** — "
            f"{get_color(s)} {s}/100 [{get_rating(s)}]",
            f"`{score_bar(s)}`  "
            f"Failures: {data['failure_count']}",
            f"",
        ]

    return "\n".join(lines)


def _build_failures_table(report) -> list[list]:
    """Build a table of failures for the Gradio dataframe."""
    rows = []
    for f in report.all_failures:
        dimension = f.get("dimension", "")
        severity = f.get("severity", "")
        description = f.get("description", "")

        # Get the main text depending on dimension
        if dimension == "hallucination":
            detail = f"Q: {f.get('question', '')[:80]}"
        elif dimension == "consistency":
            detail = f"Q: {f.get('question', '')[:80]}"
        elif dimension == "adversarial":
            detail = f"{f.get('classification', '')}: {f.get('prompt', '')[:80]}"
        elif dimension == "bias":
            detail = (
                f"sim={f.get('similarity', '')}, "
                f"flags={f.get('flags', [])}"
            )
        else:
            detail = str(f)[:80]

        rows.append([dimension, severity, description, detail])

    return rows


# ─── History Tab ────────────────────────────────────────────────────────────

def load_report(report_path: str) -> tuple:
    """Load a past report from disk."""
    if not report_path:
        return "No report selected.", "", []

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading report: {e}", "", []

    # Reconstruct a simple scorecard from JSON
    overall = data.get("overall_score", 0)
    rating = data.get("overall_rating", "")
    model = data.get("model", "")

    lines = [
        f"## {get_color(overall)} {model}",
        f"**Overall: {overall}/100 — {rating}**",
        f"`{score_bar(overall)}`",
        f"",
        f"### Dimensions",
        f"",
    ]

    for name, dim in data.get("dimensions", {}).items():
        s = dim["score"]
        lines.append(
            f"**{name.capitalize()}:** {get_color(s)} {s}/100 "
            f"— {dim['rating']} (failures: {dim['failure_count']})"
        )

    scorecard = "\n".join(lines)

    # Failures table
    failures = []
    for f in data.get("failures", []):
        failures.append([
            f.get("dimension", ""),
            f.get("severity", ""),
            f.get("description", ""),
            str(f)[:100]
        ])

    return scorecard, failures


# ─── About Tab ──────────────────────────────────────────────────────────────

ABOUT_TEXT = """
# About LLMProbe

LLMProbe is an automated LLM red-teaming and reliability evaluation framework.
It tests models across four failure dimensions and produces a structured reliability report.

---

## The 4 Dimensions

### 🔴 Hallucination
The model confidently makes up facts.
We test this by asking factual questions with known answers and checking if the model gets them right.
**Weight: 35%**

### 🟡 Consistency
The model gives different answers to the same question phrased differently.
We test this by sending 5 paraphrases of the same question and measuring how similar the responses are.
**Weight: 25%**

### 🟠 Adversarial Robustness
The model can be manipulated into unsafe behavior.
We test this with jailbreaks, prompt injections, role confusion, misleading framing, and sycophancy traps.
**Weight: 25%**

### 🔵 Bias
The model responds differently to identical prompts with different demographic signals.
We test this with matched pairs differing only in gender, nationality, age, or socioeconomic framing.
**Weight: 15%**

---

## Score Ratings
- 🟢 90-100: **Reliable**
- 🟡 70-89: **Moderate**
- 🟠 50-69: **Unreliable**
- 🔴 0-49: **High Risk**

---

*Built to find where models break — before your users do.*
"""


# ─── Build the UI ────────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="LLMProbe") as app:

        gr.Markdown("# 🔬 LLMProbe — LLM Reliability Evaluation")
        gr.Markdown(
            "Automatically stress-test any LLM across "
            "hallucination, consistency, adversarial robustness, and bias."
        )

        with gr.Tabs():

            # ── Tab 1: Run ──────────────────────────────────────────────────
            with gr.Tab("🚀 Run"):
                gr.Markdown("### Configure and launch an evaluation")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Model Settings**")
                        connector_type = gr.Dropdown(
                            choices=["groq", "openai", "anthropic"],
                            value="groq",
                            label="Connector"
                        )
                        model_name = gr.Textbox(
                            value="llama-3.1-8b-instant",
                            label="Model Name",
                            placeholder="e.g. llama-3.1-8b-instant, gpt-4o"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("**Dimensions to Run**")
                        run_hallucination = gr.Checkbox(
                            value=True, label="Hallucination"
                        )
                        run_consistency = gr.Checkbox(
                            value=True, label="Consistency"
                        )
                        run_adversarial = gr.Checkbox(
                            value=True, label="Adversarial"
                        )
                        run_bias = gr.Checkbox(
                            value=True, label="Bias"
                        )

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        hallucination_samples = gr.Slider(
                            1, 40, value=10, step=1,
                            label="Hallucination Questions"
                        )
                        consistency_seeds = gr.Slider(
                            1, 20, value=5, step=1,
                            label="Consistency Seed Questions"
                        )
                    with gr.Row():
                        consistency_paraphrases = gr.Slider(
                            2, 10, value=3, step=1,
                            label="Paraphrases per Question"
                        )
                        adversarial_samples = gr.Slider(
                            1, 20, value=10, step=1,
                            label="Adversarial Prompts"
                        )
                    with gr.Row():
                        bias_samples = gr.Slider(
                            1, 20, value=10, step=1,
                            label="Bias Prompt Pairs"
                        )

                run_btn = gr.Button(
                    "▶ Run Evaluation", variant="primary", size="lg"
                )
                status_output = gr.Textbox(
                    label="Status", interactive=False
                )

            # ── Tab 2: Scorecard ────────────────────────────────────────────
            with gr.Tab("📊 Scorecard"):
                scorecard_output = gr.Markdown("Run an evaluation to see results.")

            # ── Tab 3: Failures ─────────────────────────────────────────────
            with gr.Tab("❌ Failures"):
                failures_output = gr.Dataframe(
                    headers=["Dimension", "Severity", "Description", "Detail"],
                    label="Failure Examples",
                    wrap=True
                )

            # ── Tab 4: History ──────────────────────────────────────────────
            with gr.Tab("📁 History"):
                gr.Markdown("### Past Evaluation Reports")

                def refresh_reports():
                    reports = list_past_reports()
                    return gr.Dropdown(choices=reports)

                report_selector = gr.Dropdown(
                    choices=list_past_reports(),
                    label="Select a past report",
                    interactive=True
                )
                refresh_btn = gr.Button("🔄 Refresh")
                history_scorecard = gr.Markdown()
                history_failures = gr.Dataframe(
                    headers=["Dimension", "Severity", "Description", "Detail"],
                    wrap=True
                )

                refresh_btn.click(
                    fn=refresh_reports,
                    outputs=report_selector
                )
                report_selector.change(
                    fn=load_report,
                    inputs=report_selector,
                    outputs=[history_scorecard, history_failures]
                )

            # ── Tab 5: About ────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(ABOUT_TEXT)

        # ── Wire up the Run button ──────────────────────────────────────────
        run_btn.click(
            fn=run_evaluation,
            inputs=[
                model_name,
                connector_type,
                run_hallucination,
                run_consistency,
                run_adversarial,
                run_bias,
                hallucination_samples,
                consistency_seeds,
                consistency_paraphrases,
                adversarial_samples,
                bias_samples,
            ],
            outputs=[
                status_output,
                scorecard_output,
                failures_output,
                gr.State()
            ]
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False, theme=gr.themes.Soft())