# Pre-tokenizer pattern

`bpe.py` splits text into pieces with one regular expression, compiled with the `regex` module, before it counts any pair. Merges never cross a piece boundary. This note explains the pattern and the alternatives we considered, for Part B of the report.

## The pattern (`bpe.URDU_PATTERN`)

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)
| ?[\p{L}\p{M}‌]+
| ?\p{N}{1,3}
| ?[^\s\p{L}\p{M}\p{N}‌]+
|\s*[\r\n]+
|\s+(?!\S)
|\s+
```

The alternatives are tried in order at each position:

1. **`(?i:'s|'t|'re|'ve|'m|'ll|'d)`: English contractions.** These cover the English code-mixed into Urdu text: "it's" gives `it` + `'s`. The match ignores case, so "DON'T" gives ` DON` + `'T`. GPT-2's pattern matches only lower case and leaves `'` and `T` as two pieces. Urdu has no apostrophe contractions, so this alternative never fires inside Urdu.
2. **` ?[\p{L}\p{M}‌]+`: a word.** A word is a run of letters, combining marks and ZWNJ (U+200C), with an optional leading space. The two additions to GPT-2's ` ?\p{L}+` are what Urdu needs:
   - `\p{M}` keeps the diacritics inside the word: zabar, zer, pesh, shadd and the superscript alef. So اُردُو is one piece.
   - ZWNJ keeps compounds whole: صورت‌حال. normalize.py keeps ZWNJ only between two Arabic-script letters (docs/decisions.md, Step 4), so inside a word is the only place it can occur.

   The leading space belongs to the word that follows, as in GPT-2. So " اُردو زبان" gives ` اُردو` + ` زبان`, and BPE learns "space + word" tokens.
3. **` ?\p{N}{1,3}`: numbers.** Digits come in groups of at most three: "1234567" gives `123` + `456` + `7`. Otherwise every year, price and phone number in the news would compete for vocabulary slots. There are at most 2,220 number pieces: 1,110 digit strings (10 + 100 + 1,000), each with or without a leading space. So the vocabulary cannot fill up with numbers. normalize.py writes all digits as ASCII, so Urdu digits ۰–۹ do not appear in the corpus. "2024ء" gives `202` + `4` + `ء`: the year sign ء is a letter, so it starts a new piece.
4. **` ?[^\s\p{L}\p{M}\p{N}‌]+`: punctuation and symbols.** Everything that is not whitespace, a word character or a digit: "۔", "،", "؟", quotes, brackets, emoji. A run stays together ("۔\"", "!!!"), with an optional leading space. The Urdu full stop is its own piece, so "ہے۔" gives ` ہے` + `۔`, and BPE can learn ` ہے` once instead of a separate ` ہے۔`.
5. **`\s*[\r\n]+`: line breaks.** A run of newlines, together with any whitespace before it, is one piece. Paragraph breaks in train.txt become single tokens.
6. **`\s+(?!\S)` then `\s+`: other whitespace.** A run of spaces leaves its last space for the word that follows: "a  b" gives `a` + ` ` + ` b`. A run at the end of the text is taken whole by the last alternative.

Every character is matched by exactly one of these alternatives:
- whitespace by 5–6;
- a letter, mark or ZWNJ by 2;
- a digit by 3;
- anything else by 4, including lone surrogates and unassigned code points.

So the pieces always join back to the original text. `tests/test_bpe_reference.py` checks this on 5,000 random strings from all Unicode planes.

`<|endoftext|>` never reaches the pattern. Training and `encode` first split the text at the special token, and pre-tokenize only the pieces between.

## What GPT-2's pattern does to Urdu

The handout's GPT-2 pattern is `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`. Its letter class has no `\p{M}`, so a diacritic falls into the punctuation class and ends the word. Measured with `regex` (`tests/test_bpe_reference.py::test_gpt2_pattern_cuts_urdu_words`; run with `-s` to print):

| Input | GPT-2 pattern | Our pattern |
|---|---|---|
| اُردُو (handout's spelling, two pesh) | 5 pieces: `ا` `ُ` `رد` `ُ` `و` | `اُردُو` |
| اُردو (one pesh) | 3 pieces: `ا` `ُ` `ردو` | `اُردو` |
| " اُردو زبان" | ` ا` `ُ` `ردو` ` زبان` | ` اُردو` ` زبان` |
| صورت‌حال (with ZWNJ) | `صورت` `‌` `حال` | `صورت‌حال` |
| 2024ء | `2024` `ء` | `202` `4` `ء` |
| it's DON'T | `it` `'s` ` DON` `'` `T` | `it` `'s` ` DON` `'T` |

Unvowelled Urdu (most of the corpus) is not cut by GPT-2's pattern. Words with harakat (Makhzan, Wikipedia, religious text) and ZWNJ compounds are. Under GPT-2's pattern, BPE would also learn the bare pesh and the bare ZWNJ as separate tokens.

## Alternatives considered

- **GPT-2's pattern as is.** Rejected for the reasons in the table: it cuts words at combining marks and at ZWNJ, and its numbers have no length limit.
- **o200k_base's pattern** (GPT-4o; described from memory, and `analysis/compare_tokenizers.py` will print the exact string with tiktoken, the one place tiktoken is allowed). It already puts `\p{M}` in its letter classes, so it keeps اُردُو whole. It differs from ours in four ways:
  - It splits letters by case (upper-case runs and lower-case runs), which means nothing for Arabic script.
  - It attaches one optional leading character that is not a letter or digit (not only a space) to a word, so "(پاکستان" or "«اردو" would be one piece. That multiplies the variants of each frequent word by the brackets and quotes before it.
  - It attaches contractions to the end of the word.
  - ZWNJ (format character Cf) is outside its letter classes, so a ZWNJ compound is split.

  We kept GPT-2's simpler structure and added only `\p{M}`, ZWNJ, case-insensitive contractions, three-digit numbers and the newline rule. Every part can then be explained, and tested on Urdu, in a few lines.
- **Splitting by script** (Arabic-script runs and Latin runs as separate pieces even without a space). This would stop tokens that mix scripts, such as "PSLکا". In normalized text such words are rare, because Urdu writes a space between an English word and the following postposition. Splitting would also need a script property for every letter (`\p{Script=Arabic}`) and a rule for marks shared between scripts. Not worth the extra complexity; Part C can check whether mixed-script tokens appear among the merges.
