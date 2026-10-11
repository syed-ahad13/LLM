# Data card: Urdu corpus

> **Stub.** A script in `clean/` will generate these tables from `data/stats/`. Do not fill in numbers by hand. The handout limits this card to one page.

**Definitions.**
- MB = 10^6 bytes of UTF-8. MiB is also given.
- A word is a `str.split()` token.
- Unique words are distinct maximal runs of `[\p{L}\p{M}‌]`.

## Sources

| Source | Genre | URL | Licence / terms | Collected | Raw size | Cleaned size |
|---|---|---|---|---|---|---|
| _TBD_ | | | | | | |

## Corpus totals

Split by document: 98/1/1, seed 1337, stratified by source.

| Split | MB | MiB | Documents | Words | Unique words |
|---|---|---|---|---|---|
| train | | | | | |
| val | | | | | |
| test | | | | | |
| **total** | | | | | |

## Cleaning steps

| Step | Docs removed | Lines removed | MB removed | % of input bytes |
|---|---|---|---|---|
| 1. Extract plain text | | | | |
| 2. Normalize | | | | |
| 3. Line language filter | | | | |
| 4. Boilerplate-line removal | | | | |
| 5. Exact document dedup | | | | |
| 6. Near-dedup (optional) | | | | |

## Known gaps and biases

_TBD: genre balance, time range, regional and register coverage, residual noise._
