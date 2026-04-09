from pydantic import BaseModel, Field, model_validator
from typing import Literal


class ScoringWeights(BaseModel):
    hallucination: float = Field(default=0.35, ge=0, le=1)
    consistency: float = Field(default=0.25, ge=0, le=1)
    adversarial: float = Field(default=0.25, ge=0, le=1)
    bias: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def weights_must_sum_to_one(self):
        total = (
            self.hallucination
            + self.consistency
            + self.adversarial
            + self.bias
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total:.4f}")
        return self


class RunConfig(BaseModel):
    # Which model to probe
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Model identifier (e.g. gpt-4o, claude-3-haiku, mistralai/Mistral-7B)"
    )
    connector_type: Literal["openai", "anthropic", "huggingface"] = Field(
        default="openai",
        description="Which API/backend to use"
    )

    # Which dimensions to run
    dimensions: list[Literal["hallucination", "consistency", "adversarial", "bias"]] = Field(
        default=["hallucination", "consistency", "adversarial", "bias"],
        description="Evaluation dimensions to run"
    )

    # Per-dimension settings
    hallucination_sample_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of factual QA questions to test"
    )
    consistency_sample_size: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of seed questions for consistency testing"
    )
    consistency_paraphrases: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Number of paraphrases per seed question"
    )
    adversarial_sample_size: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Number of adversarial prompts to test"
    )
    bias_sample_size: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Number of bias prompt pairs to test"
    )

    # Scoring
    weights: ScoringWeights = Field(default_factory=ScoringWeights)

    # Output
    output_dir: str = Field(default="reports", description="Where to save reports")
    run_name: str = Field(default="default_run", description="Name for this run")