import re
import json
import regex
from collections.abc import Iterable, Iterator

from cs336_basics.common import GPT2_PRETOKENIZER_PATTERN, gpt2_bytes_to_unicode


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab  # id -> bytes
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        # ── Lookup structures ──
        # bytes -> id (reverse of vocab, for encoding)
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}

        # merge pair -> (rank, merged_bytes)
        # rank = index in merges list (lower = higher priority = created earlier)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # id -> bytes (for decoding, same as self.vocab)
        self.id_to_bytes: dict[int, bytes] = vocab

        # Build regex pattern to split on special tokens (longest match first)
        if self.special_tokens:
            escaped = [re.escape(st) for st in self.special_tokens]
            self._special_pattern = re.compile("|".join(escaped))
        else:
            self._special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from serialized vocab.json and merges.txt."""
        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        # ── Load vocab.json ──
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab: dict[str, int] = json.load(f)

        vocab: dict[int, bytes] = {
            token_id: bytes([byte_decoder[ch] for ch in unicode_str])
            for unicode_str, token_id in gpt2_vocab.items()
        }

        # If any special tokens are missing from the vocab, append them
        if special_tokens:
            for st in special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in set(vocab.values()):
                    vocab[len(vocab)] = st_bytes

        # ── Load merges.txt ──
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 2:
                    t1 = bytes([byte_decoder[ch] for ch in parts[0]])
                    t2 = bytes([byte_decoder[ch] for ch in parts[1]])
                    merges.append((t1, t2))

        return cls(vocab, merges, special_tokens)

    def _apply_merges(self, token_list: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a list of byte tokens in merge-creation order.

        Repeatedly scan for the highest-priority (earliest-created) merge
        that exists in the current token sequence and apply it, until no
        more merges are applicable.
        """
        best_rank = float("inf")
        best_pair = None
        while len(token_list) > 1:
            for i in range(len(token_list) - 1):
                pair = (token_list[i], token_list[i+1])
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair == None: break
            # if find merges appliable, apply here
            i = 0
            new_token_list : list[bytes] = []
            while i < len(token_list) - 1:
                pair = (token_list[i], token_list[i+1])
                if i < len(token_list) - 1 and pair == best_pair:
                    new_token_list.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_token_list.append(token_list[i])
                    i += 1
            token_list = new_token_list

        return token_list



    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Steps:
          1. Split text on special tokens (preserving them as individual tokens).
          2. Pre-tokenize non-special segments using GPT-2 regex.
          3. Convert each pre-token to UTF-8 bytes.
          4. Apply BPE merges in creation order.
          5. Look up each resulting token in vocab to get its ID.
        """
        if not text:
            return []

        ids: list[int] = []

        # Step 1: Split on special tokens
        if self._special_pattern:
            # Split while keeping the special tokens in the result
            parts = self._special_pattern.split(text)
            # re.split loses the delimiters; use findall + manual interleave
            specials = self._special_pattern.findall(text)

            segments: list[tuple[str, bool]] = []  # (text, is_special)
            for i, part in enumerate(parts):
                if part:
                    segments.append((part, False))
                if i < len(specials):
                    segments.append((specials[i], True))
        else:
            segments = [(text, False)]

        for segment, is_special in segments:
            if is_special:
                # Special token -> look up directly
                st_bytes = segment.encode("utf-8")
                ids.append(self.bytes_to_id[st_bytes])
                continue

            # Step 2: Pre-tokenize using GPT-2 regex
            for match in regex.finditer(GPT2_PRETOKENIZER_PATTERN, segment):
                pre_token = match.group()

                # Step 3: Convert to list of single-byte tokens
                token_list = [bytes([b]) for b in pre_token.encode("utf-8")]

                # Step 4: Apply BPE merges
                token_list = self._apply_merges(token_list)

                # Step 5: Convert bytes -> token IDs
                for token_bytes in token_list:
                    ids.append(self.bytes_to_id[token_bytes])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings, lazily yield token IDs.

        This is memory-efficient: instead of loading an entire file into
        a single string and encoding it all at once, we process one chunk
        (e.g., one line) at a time and yield its token IDs before moving
        to the next chunk.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Steps:
          1. Look up each ID in vocab to get its byte sequence.
          2. Concatenate all byte sequences.
          3. Decode the resulting bytes using UTF-8.
        """
        # Step 1: Look up each ID in vocab to get its byte sequence
        byte_sequences = [self.vocab[id] for id in ids]

        # Step 2: Concatenate all byte sequences
        concatenated_bytes = b"".join(byte_sequences)

        # Step 3: Decode the resulting bytes using UTF-8
        return concatenated_bytes.decode("utf-8", errors="replace")