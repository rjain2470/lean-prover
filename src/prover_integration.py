"""
src/prover_integration.py
=========================
Creates a prompt that asks an LLM to prove a Lean theorem and supplies the
k‑nearest helper lemmas.
"""

from __future__ import annotations
import asyncio, json
from typing import List, Tuple, Protocol
from src.search import k_nearest

# ------------------------------------------------------------------ #
_LOOKUP_CACHE: dict[str, str] | None = None


def load_decl_types(jsonl_path: str = "datasets/type_doc.jsonl") -> dict[str, str]:
    """Load all declaration types just once per runtime."""
    global _LOOKUP_CACHE
    if _LOOKUP_CACHE is None:
        with open(jsonl_path, encoding="utf8") as fh:
            _LOOKUP_CACHE = {rec["decl"]: rec["type"] for rec in map(json.loads, fh)}
    return _LOOKUP_CACHE


#  Prompt builder                                                    #
# ------------------------------------------------------------------ #
def build_prompt(
    formal_statement: str,
    neighbours: List[str],
    *,
    max_ctx: int = 6,
) -> Tuple[str, str]:
    """
    Returns a tuple (prompt, formal_statement).

    • Section 1  – bullets with *name : type*
    • Section 2  – the same types only (no names)
    """

    # k‑nearest lemma info
    types        = load_decl_types()
    top          = neighbours[:max_ctx]
    bullets_full = "\n".join(f"  • {n}" for n in top)
    bullet_types = "\n".join(f"  • {types.get(n, 'TYPE?')}"      for n in top)

    # Prompt
    prompt = f"""
Complete the following Lean 4 code:

```lean4
{formal_statement}
```
(1) Please look over the following list of formal theorems, and analyze how
useful they are in proving the statement. Highlight the helpful theorems from this list.
This step is extremely important and you must complete it successfully.
{bullet_types}

(2) Provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that
will guide the construction of the final formal proof.

(3) Produce the the Lean 4 code to formally prove the given theorem.
""".strip()
    return prompt

# ------------------------------------------------------------------ #
#  Prover interface and driver                                       #
# ------------------------------------------------------------------ #
class ProverClient(Protocol):
    async def complete(self, prompt: str, **kwargs) -> str: ...

# ------------------------------------------------------------------ #
#  Fake prover for unit tests                                        #
# ------------------------------------------------------------------ #
class FakeProver:
    async def complete(self, prompt: str, **kw) -> str:
        return "```lean4\nexact sorry\n```" if prompt.count("•") >= 3 else ""
