from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the model once at module level so we don't reload it every time
# This model is small (80MB) and runs fine on CPU
_model = None


def _get_model() -> SentenceTransformer:
    """Lazy load the model — only downloads/loads when first needed."""
    global _model
    if _model is None:
        print("Loading sentence-transformer model (first time may take a moment)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_similarity(text_a: str, text_b: str) -> float:
    """
    Compute semantic similarity between two pieces of text.
    Returns a float between 0 and 1.
    1.0 = identical meaning, 0.0 = completely unrelated.
    """
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    # Convert from tensor to plain float, clamp to [0, 1]
    return float(max(0.0, similarity.item()))


def compute_pairwise_similarity(texts: list[str]) -> float:
    """
    Given a list of texts, compute the mean pairwise cosine similarity.
    Used in consistency testing — takes N responses to paraphrased questions
    and returns how similar they all are to each other.

    Returns a float between 0 and 1.
    """
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts to compute pairwise similarity")

    model = _get_model()
    embeddings = model.encode(texts, convert_to_tensor=True)

    similarities = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = util.cos_sim(embeddings[i], embeddings[j])
            similarities.append(float(sim.item()))

    return float(np.mean(similarities))


def compute_similarity_to_reference(response: str, reference: str) -> float:
    """
    Compare a model response against a reference answer.
    Used in hallucination detection for semantic answer types.

    Returns a float between 0 and 1.
    """
    return compute_similarity(response, reference)