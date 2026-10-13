# Part C hand-picked inputs

Seven inputs for Part C question 2 (tokens per word against GPT-2 and o200k_base) and question 3 (inputs the tokenizer handles badly), plus one sentence for the normalization mini-table. No Makhzan text appears here (docs/decisions.md, 2026-10-11: Makhzan may not be quoted). Every file is passed through `normalize()` before any tokenizer sees it.

| File | Category | Source | Notes |
|---|---|---|---|
| `news.txt` | news | VOA Urdu, "پاکستان ہاکی ٹیم کے 8 کھلاڑیوں کو جرمانہ", 2012-05-29, <https://www.urduvoa.com/a/pakistan-sports-fieldhockey-india-29may12-155341065/1350315.html>; doc `own_scrape:e38fa684d64aa1e5e9b0e0d170d8f82a65e9d0c5` in `test.txt` | Copied exactly from the cleaned corpus (title line + body). VOA's own text is public domain. Being from test.txt, the tokenizer never trained on it. |
| `poetry.txt` | poetry | Mirza Ghalib, ghazal "ہزاروں خواہشیں ایسی کہ ہر خواہش پہ دم نکلے" (public domain, d. 1869) | **Typed from memory; check against a printed source (e.g. Rekhta, Diwan-e-Ghalib) before the report.** 8 of the ghazal's couplets. |
| `urdu_english.txt` | Urdu with English words | Written by us for this assignment | Tech-news style paragraph with English terms in Latin script. Invented; not a real news report and quotes no real person. |
| `roman_urdu.txt` | Roman Urdu | Written by us | Everyday Urdu in Latin script, common informal spellings. |
| `numbers_dates.txt` | numbers and dates | Written by us | Dates, times, grouped and long numbers, phone and ID-card formats (dummy numbers), decimals, percentages, ordinals, and one date in Urdu digits (normalized to ASCII). The two historical dates are 23 March 1940 and 14 August 1947. |
| `diacritics.txt` | text with diacritics | Allama Iqbal, "بچے کی دعا" (لب پہ آتی ہے دعا), first three couplets (public domain, d. 1938), and ten dictionary-style entries written by us | **Couplets typed from memory; check the words against a printed source.** All diacritics (zabar, zer, pesh, tashdeed) were added by us. |
| `names.txt` | personal names | Written by us | 25 Pakistani and 25 foreign names of public figures, in Urdu script. |
| `unnormalized.txt` | normalization mini-table only | Written by us | One sentence with Arabic kaf (U+0643), yeh (U+064A) and heh (U+0647) in place of Urdu ک, ی, ہ, generated from its normalized form, so `normalize(raw)` gives back the Urdu sentence exactly. |
