"""
src/search.py
-------------
Runtime helper for embedding queries and retrieving nearest Lean
declarations from the pre-built FAISS index.

embed(text: str) -> np.ndarray
    Single 3 072-d embedding for arbitrary text.

k_nearest(query: str, k: int = 15) -> list[tuple[str, float]]
    Returns top-k (declaration_name, cosine_distance) pairs.
    Relies on datasets/mathlib4_hnsw.faiss and its companion .names.npy
    being present on disk at import time.
"""

import os, numpy as np, faiss, openai

# obtain API key (supports both Colab and local execution)
try:
    from google.colab import userdata
    openai.api_key = userdata.get("OPENAI_API_KEY")
except ModuleNotFoundError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL       = "text-embedding-3-large"             # embedding model name
INDEX_PATH  = "datasets/mathlib4_hnsw.faiss"       # FAISS index on disk
NAMES_PATH  = "datasets/mathlib4_hnsw.names.npy"   # row-id → Lean name

index = faiss.read_index(INDEX_PATH)               # load index into RAM
names = np.load(NAMES_PATH)                        # parallel name array


def embed(text: str) -> np.ndarray:
    """Return a single 3072-dim embedding for the supplied string."""
    vec = openai.embeddings.create(
        model=MODEL,
        input=text,
        encoding_format="float"
    ).data[0].embedding
    return np.asarray(vec, dtype="float32")


def k_nearest(query: str, k: int = 15):
    """
    Retrieve the k nearest declarations to `query`, as a list of
    (declaration_name, distance) tuples.

    The distance reported is the squared Euclidean distance between
    L2-normalized vectors. This is related to cosine similarity (CS) by:
    Distance = 2 * (1 - CS). A distance of 0 means perfect similarity (CS=1).
    The distance is effectively bounded between 0 and 2 in this application.
    """
    q = embed(query).reshape(1, -1)
    faiss.normalize_L2(q) # Normalize the query vector
    D, I = index.search(q, k)
    # D contains squared Euclidean distances on L2-normalized vectors

    # Filter out results where the index is -1 (invalid results)
    valid_results = [(names[int(i)], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1]

    return valid_results


if __name__ == "__main__":
    # ----------------- EXAMPLE USAGE ----------------------------------
    # This says that if a relation r is symmetric, XrY = YrX
    k = 10
    example = k_nearest("Symmetric r → swap r = r", k)
    for name, dist in example:  # Print k nearest neighbors
        print(f"{dist:6.3f}  {name}")
    # ------------------------------------------------------------------
