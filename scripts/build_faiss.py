"""
----------------------
Command-line wrapper around src.build_index.build_hnsw.  Reads a raw
embedding matrix, builds an HNSW-Flat index, and writes
<out-prefix>.faiss plus <out-prefix>.names.npy.

Usage:
    python scripts/build_faiss.py
    python scripts/build_faiss.py --vecs my.vecs --jsonl my.jsonl --out-prefix idx
"""

#!/usr/bin/env python
import argparse, importlib, pathlib, sys

# ensure the local src package is importable when running from repo root
if "./src" not in sys.path:
    sys.path.insert(0, "./src")

from src.build_index import build_hnsw # function that builds the index

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecs",       default="datasets/vecs.f32",
                        help="raw float32 matrix produced by run_embedding.py")
    parser.add_argument("--jsonl",      default="datasets/type_doc.jsonl",
                        help="JSONL whose rows align with the matrix")
    parser.add_argument("--out-prefix", default="datasets/mathlib4_hnsw",
                        help="prefix for .faiss and .names.npy outputs")
    args = parser.parse_args()
    import src.build_index as bi
    importlib.reload(bi)
    from src.build_index import build_hnsw

    n_vecs = build_hnsw(args.vecs, args.jsonl, args.out_prefix)
    print(f"Indexed {n_vecs:,} vectors.")


if __name__ == "__main__":
    if sys.argv and sys.argv[0].endswith(("ipykernel_launcher.py", "colab_kernel_launcher.py")):
        sys.argv = [sys.argv[0]]
    main()
