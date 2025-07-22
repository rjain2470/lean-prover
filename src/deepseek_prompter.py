"""
deepseek_prompter.py
------------------
Utility function that sends a Lean 4 theorem (and a natural-language prompt)
to a DeepSeek‑Prover checkpoint and returns the generated text.

Example
-------
from deepseek_prompter import generate_proof

output = generate_proof(
    model_id="deepseek-ai/DeepSeek-Prover-V2-7B",
    formal_statement=\"\"\"import Mathlib\n\nset_option maxHeartbeats 0\n\n
      theorem demo : 1 + 1 = 2 := by\n        sorry\"\"\".strip(),
    prompt=\"\"\"Complete the following Lean 4 code:\n```lean4\n{}\n```\n\"\"\"
)
print(output[:500])
"""

from __future__ import annotations
import time, asyncio, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _load_model(model_id: str):
    """Download tokenizer + model once and return both."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",                 # GPU + CPU if needed
        torch_dtype=torch.bfloat16,        # BF16 weights
        trust_remote_code=True,
    )
    return tok, mdl

async def _async_generate(tok, mdl, chat, *, max_tokens=2048):
    """Async helper: tokenise, generate, decode."""
    input_ids = tok.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(mdl.device)

    gen_ids = await asyncio.get_event_loop().run_in_executor(
        None, lambda: mdl.generate(input_ids, max_new_tokens=max_tokens)
    )
    return tok.decode(gen_ids[0], skip_special_tokens=True)

# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
def generate_proof(
    model_id: str,
    formal_statement: str,
    prompt: str,
    *,
    max_tokens: int = 2048,
) -> str | asyncio.Future:
    tok, mdl = _load_model(model_id)
    #chat = [{"role": "user", "content": prompt.format(formal_statement)}]
    chat = [{"role": "user", "content": prompt}]

    async def _job():
        tic = time.time()
        s = await _async_generate(tok, mdl, chat, max_tokens=max_tokens)
        print(f"[DeepSeek] generation took {time.time() - tic:.1f} s")
        return s

    loop = asyncio.get_event_loop()
    if loop.is_running():
        return asyncio.create_task(_job())
    else:
        return asyncio.run(_job())
