import regex

# GPT-2 pre-tokenization pattern (requires `regex` package for Unicode properties)
GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte -> printable unicode mapping.

    Maps each of the 256 possible byte values to a unique printable
    Unicode character.  The 188 bytes that already correspond to
    printable characters (!, ", #, ..., ~, ¡, ..., ÿ minus a few)
    keep their identity; the remaining 68 are shifted to chr(256+n).
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def bytes_to_unicode_str(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    """Convert a bytes sequence to its GPT-2 unicode string representation."""
    return "".join(byte_encoder[b] for b in token_bytes)