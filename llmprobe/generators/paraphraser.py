from llmprobe.connectors.base import BaseConnector


PARAPHRASE_PROMPT = """Generate {n} different ways to ask the following question.
Each paraphrase should ask for exactly the same information but use different wording.
Return only the paraphrased questions, one per line, numbered 1 to {n}.
Do not include any explanation or the original question.

Original question: {question}"""


def generate_paraphrases(
    question: str,
    connector: BaseConnector,
    n: int = 5
) -> list[str]:
    """
    Generate n paraphrases of a question using the connector.

    Args:
        question: The original question to paraphrase
        connector: Any BaseConnector (uses Groq by default in our setup)
        n: Number of paraphrases to generate

    Returns:
        List of paraphrased questions (length may be less than n if parsing fails)
    """
    prompt = PARAPHRASE_PROMPT.format(question=question, n=n)

    response = connector.complete(prompt, temperature=0.7)
    # Use higher temperature for paraphrasing — we WANT variety here
    # unlike evaluation where we want deterministic responses

    paraphrases = _parse_numbered_list(response, n)
    return paraphrases


def _parse_numbered_list(text: str, expected: int) -> list[str]:
    """
    Parse a numbered list from the model response.

    Handles formats like:
    1. How does X work?
    1) How does X work?
    1 How does X work?
    """
    lines = text.strip().split("\n")
    paraphrases = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove numbering patterns: "1.", "1)", "1 "
        import re
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^\d+\s+", "", cleaned)
        cleaned = cleaned.strip()

        if cleaned and len(cleaned) >= 5:
            paraphrases.append(cleaned)

    return paraphrases[:expected]