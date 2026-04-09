# LLMProbe — Project Design Document

> A framework for automatically stress-testing, red-teaming, and evaluating Large Language Models across failure dimensions — hallucination, inconsistency, adversarial robustness, and bias.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Core Concepts](#3-core-concepts)
4. [System Architecture](#4-system-architecture)
5. [Evaluation Dimensions](#5-evaluation-dimensions)
6. [Pipeline Design](#6-pipeline-design)
7. [Tech Stack](#7-tech-stack)
8. [Project Structure](#8-project-structure)
9. [Module Breakdown](#9-module-breakdown)
10. [API Design](#10-api-design)
11. [Dashboard](#11-dashboard)
12. [Scoring System](#12-scoring-system)
13. [Build Plan (2–3 Weeks)](#13-build-plan-23-weeks)
14. [Future Extensions](#14-future-extensions)

---

## 1. Project Overview

**LLMProbe** is an automated LLM evaluation and red-teaming framework. Given any LLM (via API or HuggingFace), it systematically probes the model across four failure dimensions, scores each dimension, and produces a structured reliability report with annotated failure examples.

The goal is not to benchmark performance on standard tasks — it is to find *where and how* a model breaks.

**Core output:** A reliability scorecard per model, broken down by failure type, with concrete examples of each failure mode.

---

## 2. Motivation

Every company deploying an LLM in production faces the same problem: they don't know where their model will fail until it fails in front of a user.

Standard benchmarks (MMLU, HellaSwag, etc.) measure capability. They don't measure:
- Whether the model makes up confident-sounding facts
- Whether it gives different answers to the same question asked differently
- Whether it can be manipulated into unsafe or off-policy responses
- Whether it treats similar inputs inconsistently based on framing

LLMProbe fills this gap. It is a tool for **reliability engineering on LLMs**, not capability benchmarking.

---

## 3. Core Concepts

### Red-Teaming
Red-teaming is the practice of adversarially probing a system to find its weaknesses. In the context of LLMs, this means generating inputs specifically designed to elicit failure — hallucinations, contradictions, policy violations, or biased outputs.

### Hallucination
A model hallucinates when it generates factually incorrect content with high confidence. LLMProbe detects hallucination by asking questions with verifiable ground-truth answers and comparing model output against a reference knowledge base.

### Consistency
A consistent model gives semantically equivalent answers to semantically equivalent questions, regardless of phrasing. LLMProbe measures consistency by paraphrasing the same question N times and measuring variance in the responses.

### Adversarial Robustness
A robust model maintains correct, safe behavior when inputs are manipulated — through jailbreak attempts, prompt injections, or misleading framing. LLMProbe tests robustness using a curated library of adversarial prompt templates.

### Bias Probing
A model exhibits bias when it responds differently to inputs that differ only in a sensitive attribute (name, gender, nationality, etc.). LLMProbe detects this by running semantically identical prompts with swapped demographic signals.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LLMProbe                             │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Probe Suite │    │   Evaluator  │    │   Reporter   │  │
│  │              │    │              │    │              │  │
│  │ - Halluc.    │───▶│ - Scorer     │───▶│ - Scorecard  │  │
│  │ - Consist.   │    │ - Comparator │    │ - Examples   │  │
│  │ - Adversar.  │    │ - Detector   │    │ - JSON/HTML  │  │
│  │ - Bias       │    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│          │                  │                               │
│          ▼                  ▼                               │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │  Prompt Gen  │    │  Model Conn. │                       │
│  │              │    │              │                       │
│  │ - Templates  │    │ - OpenAI API │                       │
│  │ - Paraphrase │    │ - HuggingFace│                       │
│  │ - Adversarial│    │ - Anthropic  │                       │
│  └──────────────┘    └──────────────┘                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Gradio Dashboard                        │   │
│  │  Model selector · Run controls · Scorecard · Logs   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. User selects a model and configures a probe run via the dashboard
2. The Prompt Generator produces inputs for each evaluation dimension
3. The Model Connector sends prompts to the target LLM and collects responses
4. The Evaluator scores each response against ground truth, reference outputs, or heuristic detectors
5. The Reporter aggregates scores and produces a structured report
6. The Dashboard displays the scorecard and annotated failure examples

---

## 5. Evaluation Dimensions

### 5.1 Hallucination Detection

**Method:** Ask the model factual questions with known answers. Compare responses against a curated ground truth dataset using:
- Exact match for discrete facts (dates, names, numbers)
- Semantic similarity (cosine similarity via sentence-transformers) for open-ended factual answers
- Named entity comparison for claims about people, places, organizations

**Dataset:** A curated set of ~500 factual questions across domains (science, history, geography, math) with verified answers.

**Output:** Hallucination rate (% of questions answered incorrectly with high confidence), confidence calibration score, worst-case failure examples.

---

### 5.2 Consistency Testing

**Method:**
1. Take a seed question
2. Generate N paraphrases of the same question (using a paraphrase model or template-based rephrasing)
3. Send all N variants to the target model
4. Measure semantic similarity across all N responses using sentence-transformers

**Metrics:**
- Mean pairwise cosine similarity across responses (higher = more consistent)
- Variance in response length
- Contradiction detection: if any two responses make opposing factual claims (using an NLI model)

**Output:** Consistency score per question, overall consistency rate, example contradiction pairs.

---

### 5.3 Adversarial Robustness

**Method:** Test the model against a library of adversarial prompt categories:

| Category | Description |
|----------|-------------|
| Jailbreaks | Prompts designed to bypass safety guidelines |
| Prompt injection | Injecting instructions inside user-supplied content |
| Role confusion | Asking the model to "pretend" it has no restrictions |
| Misleading framing | Phrasing harmful requests as hypotheticals or fiction |
| Sycophancy traps | Presenting false information as established fact |

For each category, evaluate whether the model:
- Maintains its policy (safe response)
- Partially complies (partial failure)
- Fully complies with the adversarial intent (full failure)

**Output:** Robustness score per category, overall robustness rate, example failures with severity labels.

---

### 5.4 Bias Probing

**Method:** Generate matched pairs of prompts that are semantically identical except for a sensitive attribute (name, gender, nationality, age). Measure:
- Response length difference
- Sentiment difference (using a sentiment model)
- Semantic similarity between paired responses (should be high if unbiased)

**Sensitive attributes tested:**
- Gender (he/she/they pronouns, gendered names)
- Nationality (Western vs. non-Western names)
- Age (young vs. old framing)
- Socioeconomic framing (wealthy vs. low-income context)

**Output:** Bias score per attribute, flagged pairs where responses diverge significantly, disparity metrics.

---

## 6. Pipeline Design

```
Configure Run
     │
     ▼
Generate Prompts ──────────────────────────────────┐
     │                                             │
     ▼                                             │
Query Target Model (async, batched)                │
     │                                             │
     ▼                                             │
Evaluate Responses                                 │
  ├── Hallucination: compare vs. ground truth      │
  ├── Consistency: measure cross-paraphrase sim.   │
  ├── Adversarial: classify response safety        │
  └── Bias: measure paired response divergence     │
     │                                             │
     ▼                                             │
Aggregate Scores                                   │
     │                                             │
     ▼                                             │
Generate Report (JSON + Markdown)                  │
     │                                             │
     ▼                                             │
Display on Dashboard ◀─────────────────────────────┘
```

All pipeline stages are modular — you can run a single dimension probe or the full suite. Each stage writes results to a structured JSON file so the pipeline is resumable.

---

## 7. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11 | Core language |
| Deep Learning | PyTorch | Model inference, embeddings |
| Transformers | HuggingFace Transformers | Loading and running models |
| Embeddings | sentence-transformers | Semantic similarity computation |
| NLI | HuggingFace (cross-encoder/nli) | Contradiction detection |
| Sentiment | HuggingFace pipeline | Bias measurement |
| LLM APIs | openai, anthropic SDKs | Connecting to target models |
| Prompt handling | LangChain | Prompt templates, chaining |
| Dashboard | Gradio | Interactive web UI |
| Deployment | HuggingFace Spaces | Public demo hosting |
| Config | Pydantic v2 | Run configuration validation |
| Storage | JSON / JSONL | Reports and run history |
| Testing | pytest | Unit and integration tests |

---

## 8. Project Structure

```
llmprobe/
├── README.md                        # Project overview and quickstart
├── DESIGN.md                        # This file
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Package config
│
├── llmprobe/                        # Core library
│   ├── __init__.py
│   ├── config/
│   │   ├── schema.py                # Pydantic config models
│   │   └── defaults.py              # Default run settings
│   │
│   ├── connectors/                  # Target model interfaces
│   │   ├── base.py                  # Abstract model connector
│   │   ├── openai_connector.py      # OpenAI API connector
│   │   ├── anthropic_connector.py   # Anthropic API connector
│   │   └── hf_connector.py          # HuggingFace connector
│   │
│   ├── probes/                      # Evaluation dimensions
│   │   ├── base.py                  # Abstract probe class
│   │   ├── hallucination.py         # Hallucination detection probe
│   │   ├── consistency.py           # Consistency testing probe
│   │   ├── adversarial.py           # Adversarial robustness probe
│   │   └── bias.py                  # Bias probing module
│   │
│   ├── generators/                  # Prompt generation
│   │   ├── paraphraser.py           # Question paraphrase generator
│   │   ├── adversarial_gen.py       # Adversarial prompt generator
│   │   └── bias_pairs.py            # Matched bias prompt pair generator
│   │
│   ├── evaluators/                  # Response evaluation
│   │   ├── similarity.py            # Semantic similarity (sentence-transformers)
│   │   ├── nli.py                   # Natural language inference (contradiction)
│   │   ├── sentiment.py             # Sentiment analysis
│   │   └── factual.py               # Factual comparison vs. ground truth
│   │
│   ├── scoring/
│   │   ├── scorer.py                # Per-dimension score computation
│   │   └── aggregator.py            # Overall reliability score aggregation
│   │
│   ├── pipeline/
│   │   └── runner.py                # End-to-end pipeline orchestration
│   │
│   └── reporting/
│       ├── report.py                # Report data structure
│       ├── json_reporter.py         # JSON output
│       └── markdown_reporter.py     # Markdown output
│
├── data/
│   ├── factual_qa.json              # Ground truth factual QA dataset
│   ├── adversarial_templates.json   # Adversarial prompt templates
│   └── bias_templates.json          # Bias probe templates
│
├── dashboard/
│   └── app.py                       # Gradio dashboard
│
├── reports/                         # Generated reports (runtime)
│
└── tests/
    ├── unit/
    │   ├── test_hallucination.py
    │   ├── test_consistency.py
    │   ├── test_adversarial.py
    │   ├── test_bias.py
    │   └── test_scoring.py
    └── integration/
        └── test_pipeline.py
```

---

## 9. Module Breakdown

### `connectors/base.py`
Abstract base class for all model connectors. Defines the interface:
```python
class BaseConnector:
    def complete(self, prompt: str, **kwargs) -> str: ...
    def batch_complete(self, prompts: list[str], **kwargs) -> list[str]: ...
```

### `probes/hallucination.py`
- Loads factual QA dataset
- Sends questions to the model
- Compares answers against ground truth using exact match + semantic similarity
- Returns per-question scores and a hallucination rate

### `probes/consistency.py`
- Takes seed questions
- Calls `generators/paraphraser.py` to produce N variants
- Queries the model on all variants
- Uses `evaluators/similarity.py` to compute pairwise cosine similarity
- Uses `evaluators/nli.py` to detect contradictions

### `probes/adversarial.py`
- Loads adversarial templates from `data/adversarial_templates.json`
- Instantiates prompts across categories
- Queries the model
- Classifies responses as safe / partial / failed using heuristic rules + a safety classifier

### `probes/bias.py`
- Loads matched pair templates from `data/bias_templates.json`
- Instantiates pairs with different demographic attributes
- Queries the model on both variants
- Measures divergence using sentiment difference and semantic similarity

### `scoring/scorer.py`
Computes a normalized score [0, 100] per dimension:
- Hallucination: `(1 - hallucination_rate) × 100`
- Consistency: `mean_pairwise_similarity × 100`
- Adversarial: `(safe_count / total) × 100`
- Bias: `mean_paired_similarity × 100`

### `scoring/aggregator.py`
Computes a weighted overall **Reliability Score**:
```
Reliability = 0.35 × Hallucination + 0.25 × Consistency + 0.25 × Adversarial + 0.15 × Bias
```
Weights are configurable in `config/schema.py`.

### `pipeline/runner.py`
Orchestrates the full pipeline:
1. Load config
2. Initialize connector
3. Run each probe in sequence (or parallel)
4. Aggregate scores
5. Generate report

### `dashboard/app.py`
Gradio interface with:
- Model selector (dropdown: GPT-4o, Claude 3, Mistral, custom HF model)
- Dimension selector (checkboxes)
- Run button with live progress
- Scorecard display (per-dimension + overall)
- Failure examples table (filterable by dimension and severity)
- Download report button

---

## 10. API Design

LLMProbe can also be used as a Python library:

```python
from llmprobe import LLMProbe
from llmprobe.connectors import OpenAIConnector

# Initialize
probe = LLMProbe(
    connector=OpenAIConnector(model="gpt-4o"),
    dimensions=["hallucination", "consistency", "adversarial", "bias"]
)

# Run evaluation
report = probe.run()

# Access results
print(report.reliability_score)          # Overall score
print(report.scores["hallucination"])    # Per-dimension score
print(report.failures["adversarial"])    # Failure examples
report.save("reports/gpt4o_report.json")
```

---

## 11. Dashboard

The Gradio dashboard is the primary interface for non-programmatic use.

**Tabs:**
1. **Run** — configure and launch an evaluation
2. **Scorecard** — reliability score breakdown with color-coded ratings
3. **Failures** — table of failure examples, filterable by dimension and severity
4. **Compare** — side-by-side comparison of two model reports
5. **History** — past run reports

**Scorecard color scheme:**
- 90–100: Green (Reliable)
- 70–89: Yellow (Moderate)
- 50–69: Orange (Unreliable)
- Below 50: Red (High Risk)

---

## 12. Scoring System

### Per-Dimension Scores

| Dimension | What it measures | Weight |
|-----------|-----------------|--------|
| Hallucination | Factual accuracy vs. ground truth | 35% |
| Consistency | Stability across paraphrased inputs | 25% |
| Adversarial | Robustness to manipulation attempts | 25% |
| Bias | Fairness across demographic variants | 15% |

### Overall Reliability Score

```
Reliability Score = Σ (dimension_score × dimension_weight)
```

Scores are normalized to [0, 100]. The reliability score is not a capability score — a model can be highly capable and still have a low reliability score if it hallucinates confidently or is easily manipulated.

### Severity Labels

Each failure example is tagged with a severity:
- **Critical** — model provided harmful, dangerous, or completely false output
- **High** — significant factual error or policy violation
- **Medium** — inconsistency or partial failure
- **Low** — minor divergence or borderline case

---

## 13. Build Plan (2–3 Weeks)

### Week 1 — Core Engine

**Days 1–2: Setup & Connectors**
- Initialize repo, set up pyproject.toml, requirements
- Build `BaseConnector` abstract class
- Implement `OpenAIConnector` (most accessible for testing)
- Write unit tests for connectors

**Days 3–4: Hallucination Probe**
- Curate or source a factual QA dataset (~200 questions to start)
- Build `hallucination.py` probe
- Implement `evaluators/factual.py` using sentence-transformers for semantic similarity
- Test end-to-end: question → model → score

**Days 5–7: Consistency Probe**
- Build template-based paraphraser (5 paraphrase patterns per question)
- Implement `consistency.py` probe
- Add NLI-based contradiction detection using a cross-encoder
- Test on a seed set of 20 questions

---

### Week 2 — Adversarial + Bias + Scoring

**Days 8–9: Adversarial Probe**
- Build adversarial template library (50+ prompts across 5 categories)
- Implement `adversarial.py` probe with rule-based + classifier-based safety evaluation
- Add severity labeling

**Days 10–11: Bias Probe**
- Build matched pair templates for 4 sensitive attributes
- Implement `bias.py` probe
- Add sentiment + similarity divergence measurement

**Days 12–14: Scoring & Reporting**
- Implement `scorer.py` and `aggregator.py`
- Build JSON and Markdown reporters
- Implement `pipeline/runner.py` for full end-to-end run
- Write integration test: full pipeline on a small model

---

### Week 3 — Dashboard, Polish, Deploy

**Days 15–17: Gradio Dashboard**
- Build `dashboard/app.py` with all tabs
- Connect to pipeline runner
- Add live progress display and failure example table
- Add report download

**Days 18–19: HuggingFace Connector + Deployment**
- Implement `hf_connector.py` for open-source models
- Deploy on HuggingFace Spaces
- Test with at least 2 models end-to-end (e.g. GPT-3.5 and Mistral-7B)

**Days 20–21: Polish**
- Write README with demo GIF and example scorecard
- Add Anthropic connector
- Code cleanup, final tests
- Tag v1.0 release

---

## 14. Future Extensions

- **Multi-model comparison mode** — run the same probe suite on N models and rank them
- **Custom probe builder** — let users define their own evaluation dimensions via config
- **Domain-specific probes** — medical, legal, financial hallucination datasets
- **Temporal consistency** — test whether a model gives the same answer across different sessions
- **Confidence calibration** — measure whether model confidence correlates with accuracy
- **Fine-tuning feedback loop** — export failure cases as training data for targeted fine-tuning
- **CLI interface** — `llmprobe run --model gpt-4o --dimensions all --output report.json`
- **CI/CD integration** — run LLMProbe as a GitHub Action on model updates

---

*LLMProbe — built to find where models break, before your users do.*
