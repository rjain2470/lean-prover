from src.search import embed as _embed_func, k_nearest
from src.prover_integration import build_prompt, load_decl_types
from src.embed import embed_jsonl_to_vecs

embed = _embed_func

__version__ = "0.1.0"

__all__ = [
    "embed",
    "k_nearest",
    "build_prompt",
    "load_decl_types",
    "embed_jsonl_to_vecs",
]


