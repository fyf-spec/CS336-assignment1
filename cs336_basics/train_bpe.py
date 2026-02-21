import os
import re
import json
import multiprocessing
from collections import Counter, defaultdict
from typing import BinaryIO

import regex

from cs336_basics.common import (
    GPT2_PRETOKENIZER_PATTERN,
    gpt2_bytes_to_unicode,
    bytes_to_unicode_str,
)

# Special token used as document delimiter
_ENDOFTEXT = "<|" + "endoftext" + "|>"


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def remove_special_tokens(
    chunk: str,
    special_tokens: list[str],
) -> list[str]:
    """Split chunk on special tokens, returning segments without the tokens.
    This prevents BPE merges across document boundaries."""
    if not special_tokens:
        return [chunk]
    pattern = "|".join(re.escape(token) for token in special_tokens)
    segments = re.split(pattern, chunk)
    return [seg for seg in segments if seg]


def pretokenize_chunk(args: tuple) -> Counter:
    """Pre-tokenize a single chunk: remove special tokens, apply GPT-2 regex,
    and return counts of each pre-token (as tuple of single bytes)."""
    chunk_text, special_tokens = args
    # Normalize Windows line endings for cross-platform consistency
    chunk_text = chunk_text.replace("\r\n", "\n")
    counts: Counter = Counter()
    segments = remove_special_tokens(chunk_text, special_tokens)
    for segment in segments:
        for match in regex.finditer(GPT2_PRETOKENIZER_PATTERN, segment):
            token_str = match.group()
            token_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
            counts[token_bytes] += 1
    return counts


def _get_pair_counts(pre_token_counts: dict[tuple, int]):
    """Build pair frequency counts and a reverse index from pairs to pre-tokens.

    Returns:
        pair_counts: Counter mapping (token_a, token_b) -> total frequency
        pair_to_pretokens: dict mapping (token_a, token_b) -> set of pre-token keys
    """
    pair_counts = Counter()
    pair_to_pretokens = defaultdict(set)
    for pre_token, count in pre_token_counts.items():
        for i in range(len(pre_token) - 1):
            pair = (pre_token[i], pre_token[i + 1])
            pair_counts[pair] += count
            pair_to_pretokens[pair].add(pre_token)
    return pair_counts, pair_to_pretokens


def _merge_pair(
    pre_token: tuple,
    pair: tuple,
    merged: bytes,
) -> tuple:
    """Merge all occurrences of `pair` in `pre_token` into `merged`."""
    new_token = []
    i = 0
    while i < len(pre_token):
        if i < len(pre_token) - 1 and pre_token[i] == pair[0] and pre_token[i + 1] == pair[1]:
            new_token.append(merged)
            i += 2
        else:
            new_token.append(pre_token[i])
            i += 1
    return tuple(new_token)


def save_vocab_and_merges(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    """Save vocab and merges in GPT-2 unicode format."""
    byte_encoder = gpt2_bytes_to_unicode()
    # convert vocab:dict[int, bytes] to dict[str, int]
    gpt2_vocab = {bytes_to_unicode_str(b, byte_encoder): v for v, b in vocab.items()}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(gpt2_vocab, f, ensure_ascii=False)

    # convert merges to "token1 token2" per row
    with open(merges_path, "w", encoding="utf-8") as f:
        for t1, t2 in merges:
            s1 = bytes_to_unicode_str(t1, byte_encoder)
            s2 = bytes_to_unicode_str(t2, byte_encoder)
            f.write(f"{s1} {s2}\n")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    vocab_path : str = 'vocab.json',
    merges_path : str = 'merges.txt'
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: Path to training text file.
        vocab_size: Max final vocabulary size (256 byte tokens + special tokens + merges).
        special_tokens: Special tokens to add to vocabulary (not merged).
        num_processes: Number of parallel processes for pre-tokenization.

    Returns:
        vocab: dict mapping token ID -> token bytes
        merges: list of (token1_bytes, token2_bytes) in order of creation
    """
    if num_processes is None:
        num_processes = os.cpu_count() or 4

    # ── Step 1: Parallel pre-tokenization ──────────────────────────────
    endoftext_bytes = _ENDOFTEXT.encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, endoftext_bytes)

        # Read all chunks
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append((chunk_text, special_tokens))

    # Parallel pre-tokenization
    with multiprocessing.Pool(num_processes) as pool:
        chunk_counts = pool.map(pretokenize_chunk, chunks)

    # Merge all chunk counts into one
    pre_token_counts: dict[tuple, int] = Counter()
    for cc in chunk_counts:
        pre_token_counts.update(cc)

    # ── Step 2: BPE merge loop (optimized with incremental updates) ────
    num_merges = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # Build initial pair counts and reverse index
    pair_counts, pair_to_pretokens = _get_pair_counts(pre_token_counts)

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Find the most frequent pair (break ties by lexicographic order)
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        if pair_counts[best_pair] < 1:
            break

        merged = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        # Get all pre-tokens that contain this pair
        affected_pretokens = list(pair_to_pretokens.pop(best_pair, set()))
        del pair_counts[best_pair]

        for pre_token in affected_pretokens:
            count = pre_token_counts.get(pre_token, 0)
            if count == 0:
                continue

            # Remove old pair counts for this pre-token
            for i in range(len(pre_token) - 1):
                old_pair = (pre_token[i], pre_token[i + 1])
                if old_pair == best_pair:
                    continue
                pair_counts[old_pair] -= count
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                pair_to_pretokens[old_pair].discard(pre_token)
                if not pair_to_pretokens[old_pair]:
                    del pair_to_pretokens[old_pair]

            # Create new pre-token with merge applied
            new_pre_token = _merge_pair(pre_token, best_pair, merged)

            # Remove old pre-token, add new one
            del pre_token_counts[pre_token]
            pre_token_counts[new_pre_token] = pre_token_counts.get(new_pre_token, 0) + count

            # Add new pair counts for the new pre-token
            for i in range(len(new_pre_token) - 1):
                new_pair = (new_pre_token[i], new_pre_token[i + 1])
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + count
                pair_to_pretokens[new_pair].add(new_pre_token)

    # ── Step 3: Build vocabulary ───────────────────────────────────────
    vocab: dict[int, bytes] = {}

    # First: special tokens get IDs 0, 1, 2, ...
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    # Then: 256 byte tokens
    for b in range(256):
        vocab[len(special_tokens) + b] = bytes([b])

    # Then: merged tokens in order of creation
    for i, (t1, t2) in enumerate(merges):
        vocab[len(special_tokens) + 256 + i] = t1 + t2

    # ── Step 4: Save vocabulary ───────────────────────────────────────
    save_vocab_and_merges(vocab, merges, vocab_path, merges_path)
    return vocab, merges
