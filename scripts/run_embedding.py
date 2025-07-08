"""
------------------------
CLI / notebook script that calls src.embed.embed_jsonl_to_vecs to convert
datasets/type_doc.jsonl into datasets/vecs.f32.  Handles batching,
token-budget throttling, and prints a completion summary.

Usage:
    python scripts/run_embedding.py
    # or with flags
    python scripts/run_embedding.py --jsonl custom.jsonl --vecs out.f32
"""

#!/usr/bin/env python
import os, json, asyncio, aiofiles, numpy as np, tiktoken, openai
from tqdm.asyncio import tqdm

# file paths and model parameters
JSONL_PATH = "datasets/type_doc.jsonl"
VEC_PATH   = "datasets/vecs.f32"
MODEL      = "text-embedding-3-large"            # 3 072-dim output
DIM        = 3_072
BATCH_ROWS = 2_048                               # max rows per API call
TPM_LIMIT  = 350_000                             # tokens allowed per minute

# obtain API key (works in Colab or locally)
try:
    from google.colab import userdata
    openai.api_key = userdata.get("OPENAI_API_KEY")
except ModuleNotFoundError:
    openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.AsyncOpenAI(api_key=openai.api_key)

# count input rows so we can allocate the output mem-map
n_rows = sum(1 for _ in open(JSONL_PATH, encoding="utf8"))
print(f"Counting rows in JSONL … {n_rows:,}")

# tokenizer for token-budget accounting
ENC = tiktoken.encoding_for_model(MODEL)

# memory-mapped output matrix
vecs = np.memmap(VEC_PATH, dtype="float32", mode="w+", shape=(n_rows, DIM))

# token-rate bookkeeping variables
sent_tokens  = 0
window_start = asyncio.get_event_loop().time()


async def throttle(token_count: int) -> None:
    """Pause just long enough so we stay below TPM_LIMIT."""
    global sent_tokens, window_start
    now = asyncio.get_event_loop().time()
    if now - window_start >= 60:
        sent_tokens, window_start = 0, now
    if sent_tokens + token_count > TPM_LIMIT:
        await asyncio.sleep(60 - (now - window_start))
        sent_tokens, window_start = 0, asyncio.get_event_loop().time()
    sent_tokens += token_count


async def embed(batch: list[str]) -> list[list[float]]:
    """Return embeddings for a batch; replace empty lines with zero vectors."""
    valid = [t for t in batch if t]
    if not valid:
        return [np.zeros(DIM, dtype="float32")] * len(batch)

    token_count = sum(len(ENC.encode(t)) for t in valid)
    await throttle(token_count)

    resp = await client.embeddings.create(
        model=MODEL,
        input=valid,
        encoding_format="float"
    )

    # re-insert zero vectors where empty strings were removed
    data_iter = iter(resp.data)
    out = []
    for t in batch:
        out.append(next(data_iter).embedding if t else np.zeros(DIM, dtype="float32"))
    return out


async def main() -> None:
    """Stream the JSONL file, embed each batch, and write vecs.f32."""
    row, buffer = 0, []
    async with aiofiles.open(JSONL_PATH, encoding="utf8") as fh:
        async for line in tqdm(fh, total=n_rows, unit="row"):
            buffer.append(json.loads(line)["text"])
            if len(buffer) == BATCH_ROWS:
                vecs[row:row + BATCH_ROWS] = await embed(buffer)
                row += BATCH_ROWS
                buffer.clear()
        if buffer:                                   # final (short) batch
            vecs[row:row + len(buffer)] = await embed(buffer)

    vecs.flush()
    size_mb = vecs.nbytes / 1e6
    print(f"Embedding complete. Wrote {VEC_PATH} ({size_mb:.1f} MB)")


# run from the command line
if __name__ == "__main__":
    asyncio.run(main())
