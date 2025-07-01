"""
------------------
Build an HNSW-Flat FAISS index from an on-disk embedding matrix and create
a parallel NumPy array of declaration names.

build_hnsw(
    vec_path: str,
    jsonl_path: str,
    out_prefix: str,
    dim: int = 3072,
    chunk: int = 4096,
) -> int

Args
    vec_path     path to raw float32 matrix (N × dim)
    jsonl_path   JSONL file that produced the matrix; supplies `"decl"` names
    out_prefix   path prefix; writes <prefix>.faiss and <prefix>.names.npy
    dim          embedding dimension
    chunk        rows per add() call when feeding FAISS

Returns
    number of vectors indexed
"""

import os, json, pathlib, faiss, numpy as np
from tqdm import tqdm

def build_hnsw(
    vec_path: str,           # float32 matrix (N × dim)
    jsonl_path: str,         # JSONL file aligned with vec_path
    out_prefix: str,         # prefix for output files
    dim: int = 3072,         # embedding dimension
    chunk: int = 4096        # rows to load per FAISS add() call
) -> int:
    """Create an HNSW-Flat index and a matching name array; return row count."""

    bytes_total = os.path.getsize(vec_path) # total bytes in the matrix
    n_rows = bytes_total // (dim * 4)

    # memory-map the entire vector matrix
    vecs = np.memmap(vec_path, dtype="float32", mode="r", shape=(n_rows, dim))

    # initialise an HNSW index with 32 neighbours per node
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200

    # add the vectors to FAISS in contiguous blocks
    for start in tqdm(range(0, n_rows, chunk), unit="vec"):
        end = min(start + chunk, n_rows)
        index.add(np.ascontiguousarray(vecs[start:end]))

    # write the FAISS index to disk
    out_path = pathlib.Path(out_prefix).with_suffix(".faiss")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))

    # extract declaration names from JSONL and save parallel NumPy array
    names = [json.loads(line)["decl"] for line in open(jsonl_path, "r")]
    np.save(out_path.with_suffix(".names.npy"), np.array(names))

    return n_rows
