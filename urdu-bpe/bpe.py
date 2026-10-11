"""Byte-level BPE tokenizer written from scratch: class BPETokenizer.

Spec (CLAUDE.md, "Tokenizer spec"):
- IDs 0-255 are bytes; <|endoftext|> is ID 256 and never split or merged.
- 49,743 merges with IDs 257-49,999, in the order learned.
- Regex pre-tokenization keeps Urdu words whole (\\p{M} and ZWNJ stay inside
  words); a leading space belongs to the following word.
- Ties go to the greater pair, compared as (bytes, bytes), never by ID.
- decode(encode(s)) == s for every string. The tokenizer never normalizes.
- A reference trainer (full recount per merge) and a fast trainer (word
  counts, pair -> words index, incremental updates) must learn identical
  merges on the 1 MB sample.

Not implemented yet (Step 0 scaffold).
"""
