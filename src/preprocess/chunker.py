import re
from math import ceil
from typing import Callable, List, Dict, Optional

SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[\.\?\!\n])\s+')

def default_token_count_estimate(text: str, avg_chars_per_token: float = 4.0) -> int:
    """
    Cheap heuristic: average number of characters per token. Not exact
    """
    if not text:
        return 0
    
    chars = len(text)
    return max(1, int(ceil(chars / avg_chars_per_token)))

def try_import_tiktoken():
    """
    If tiktoken is available, return tokenizer function
    Otherwise return None
    """
    try:
        import tiktoken
    except ImportError:
        return None
    
    def tokenizer_count(text: str, schema: str = "cl100k_base"):
        enc = tiktoken.get_encoding(schema)
        return len(enc.encode(text))
    
    return tokenizer_count

def split_on_sentence_boundary(text: str) -> List[str]:
    """
    Split text into sentences/blocks while preserving punctuation endings.
    Falls back to whitespace split if regex finds nothing
    """
    parts = SENTENCE_BOUNDARY_RE.split(text)
    if len(parts) == 1:
        # fallback; split on commas/semicolons or whitespace chunks to avoid huge pieces
        parts = re.split(r'(?<=[,;])\s+|\s{2,}', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_transcript(
    transcript: str,
    chunk_tokens: int = 1200,
    overlap_tokens: int = 200,
    tokenizer_fn: Optional[Callable[[str], int]] = None,
    avg_chars_per_token: float = 4.0
) -> List[Dict]:
    """
    transcript: long string separated by '\n'
    Returns: list of chunk dicts with metadata
    """
    if tokenizer_fn is None:
        tokenizer_fn = lambda t: default_token_count_estimate(t, avg_chars_per_token)

    lines = [line.strip() for line in transcript.split("\n") if line.strip()]

    chunks: List[Dict] = []
    buffer_lines: List[str] = []
    buffer_token_est = 0
    chunk_id = 0

    def emit_chunk():
        nonlocal chunk_id, buffer_lines, buffer_token_est
        if not buffer_lines:
            return None
        chunk_text = " ".join(buffer_lines).strip()
        chunk = {
            "id": chunk_id,
            "text": chunk_text,
            "token_estimate": tokenizer_fn(chunk_text),
        }
        chunk_id += 1

        # Prepare overlap for next chunk
        if overlap_tokens > 0:
            sentences = split_on_sentence_boundary(chunk_text)
            keep: List[str] =[]
            keep_est = 0
            for s in reversed(sentences):
                keep_est += tokenizer_fn(s)
                keep.insert(0, s)
                if keep_est >= overlap_tokens:
                    break

            buffer_lines = keep
            buffer_token_est = tokenizer_fn(" ".join(buffer_lines)) if buffer_lines else 0

        else:
            buffer_lines = []
            buffer_token_est = 0

        return chunk
    
    for line in lines:
        line_tokens = tokenizer_fn(line)

        # handles extremely long single lines
        if line_tokens >= chunk_tokens:
            sentences = split_on_sentence_boundary(line)
            for s in sentences:
                s_tokens = tokenizer_fn(s)
                buffer_lines.append(s)
                buffer_token_est += s_tokens
                if buffer_token_est >= chunk_tokens:
                    c = emit_chunk()
                    if c:
                        chunks.append(c)

            continue

        buffer_lines.append(line)
        buffer_token_est += line_tokens

        if buffer_token_est >= chunk_tokens:
            c = emit_chunk()
            if c:
                chunks.append(c)

    # flush remainder
    if buffer_lines:
        c = emit_chunk()
        if c:
            chunks.append(c)

    return chunks

if __name__ == "__main__":
    # example_txt = """
    # I chose Luis Alberto Urrea's “Water Museum” (2016) audiobook as my current “read” because of its genre and intriguing cover art. The genre — magical realism, the cover — a snake on a cracked desert soil. Even lettering was stylized to look dry like the soil. Simple, yet effective. It immediately conveys something full of tension, mysterious, perhaps even dangerous. Before a single word was spoken, the cover made a promise. That promise pulled me in.

    # Between 2015 and 2020, sales of audiobooks grew by 157%. Audiobooks have become a mass-market medium, with rich catalogs and broad listener adoption. I can attest to that — Audible became one of my most frequently used apps.

    # The explosion of the audiobook market in the mid 2010s led to the creation of a new type and design of book covers. Book covers that were called forth by a more complex technological landscape, habitual multitasking and preference for convenience, and ever shrinking attention spans.
    # """
    with open("./test_txt.txt") as file:
        content = file.read()
    chunked = chunk_transcript(content, tokenizer_fn=try_import_tiktoken())

    print(chunked)
