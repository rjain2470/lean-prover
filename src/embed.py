"""
------------
Stream a JSONL corpus through the OpenAI embedding endpoint and write the
resulting float32 matrix to disk in raw form.

embed_jsonl_to_vecs(
    client: openai.AsyncOpenAI,
    jsonl_path: str,
    vec_path: str,
    model: str = "text-embedding-3-large",
    batch_rows: int = 2048,
) -> tuple[int, int]

Args
    client      already-configured AsyncOpenAI client
    jsonl_path  input file containing one JSON object per line with a "text" field
    vec_path    output file; a mem-mapped N × dim float32 matrix is created here
    model       embedding model name (controls dimension)
    batch_rows  rows per API call (OpenAI hard-limit 2 048)

Returns
    (row_count, dim) written to vec_path
"""

import os, json, asyncio, openai, aiofiles, numpy as np, tiktoken
from tqdm.asyncio import tqdm as atqdm

TPM_LIMIT   = 350_000 # token limit per minute
sent_tokens = 0 # tokens sent in current minute
window_start = asyncio.get_event_loop().time()


async def throttle(tok_cnt: int) -> None:
    """Delay the coroutine so we never exceed the 60-s TPM window."""
    global sent_tokens, window_start
    now = asyncio.get_event_loop().time()
    if now - window_start >= 60:
        sent_tokens, window_start = 0, now
    if sent_tokens + tok_cnt > TPM_LIMIT:
        await asyncio.sleep(60 - (now - window_start))
        sent_tokens, window_start = 0, asyncio.get_event_loop().time()
    sent_tokens += tok_cnt


async def embed_jsonl_to_vecs(
    client: openai.AsyncOpenAI,  # OpenAI async client
    jsonl_path: str, # input JSONL with `"text"`
    vec_path: str, # output .f32 matrix
    model: str = "text-embedding-3-large",
    batch_rows: int = 2_048,
) -> tuple[int, int]:
    """
    Stream `jsonl_path` through the embedding endpoint and write a raw
    float32 matrix to `vec_path`.  Returns (row_count, dim).
    """
    enc   = tiktoken.encoding_for_model(model)     # OpenAI tokenizer
    n_rows = sum(1 for _ in open(jsonl_path, "r")) # total lines to embed
    dim   = 3_072 if "large" in model else 1_536   # dimensionality
    vecs  = np.memmap(vec_path, "float32", "w+", shape=(n_rows, dim))

    async def _embed(batch: list[str]) -> list[list[float]]:
        tok_cnt = sum(len(enc.encode(t)) for t in batch)
        await throttle(tok_cnt)
        resp = await client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )
        return [item.embedding for item in resp.data]

    row, buf = 0, []
    async with aiofiles.open(jsonl_path, "r", encoding="utf8") as fh:
        async for line in atqdm(fh, total=n_rows, unit="row"):
            buf.append(json.loads(line)["text"])
            if len(buf) == batch_rows:
                vecs[row : row + batch_rows] = await _embed(buf)
                row += batch_rows
                buf.clear()
        if buf:
            vecs[row : row + len(buf)] = await _embed(buf)

    vecs.flush()
    return n_rows, dim
