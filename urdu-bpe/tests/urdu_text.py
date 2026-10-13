"""Deterministic Urdu-like text for the tokenizer tests.

Tests cannot read data/ (gitignored) and corpus text must not be committed, so
the tests train on sentences drawn from the word list below, written for these
tests. Words are drawn with Zipf-like weights (earlier words more often), so
BPE learns real structure: two-byte letters first, then frequent words with
their leading space. The list covers diacritics, hamza and madda letters, ZWNJ
compounds, English code-mixing, digits and Urdu punctuation. Documents are
separated by <|endoftext|>, as in train.txt.
"""

import random

ENDOFTEXT = "<|endoftext|>"

WORDS = (
    "کے میں کی ہے اور کا سے کو نے یہ ہیں پر بھی ایک کہ اس وہ تھا جو کر ان ہو گیا "
    "نہیں لیے تھی ساتھ تو کیا گئی اپنے بعد تھے کچھ جب ہوں اب سب بہت آپ آج کئی "
    "پاکستان حکومت شہر ملک لوگ وقت سال دن رات پانی کتاب زبان اردو تعلیم اسکول بچے "
    "خبر کھیل ٹیم میچ بارش موسم دریا پہاڑ بازار قیمت روپے صدر وزیر اعظم عدالت فیصلہ "
    "قانون صحت ہسپتال ڈاکٹر مریض شاعر غزل ادب تاریخ صدی جنگ امن دنیا علاقے صوبے "
    "لاہور کراچی پشاور کوئٹہ ملتان اسلام آباد کرتے کرتی کہا بتایا دیکھا لکھا پڑھا "
    "جانا آنا رہا رہی ملا دیا لیا چاہیے سکتا سکتی گئے جائے سوال جواب مؤقف نقطۂ "
    "بڑا بڑی چھوٹا نیا نئی پرانا اچھا خوبصورت مشکل آسان اہم قومی عالمی ٹھیک ڈر "
    "اُردُو مُعَلِّم عِلم اَمن کِتاب مُحَبَّت صورت‌حال دل‌چسپ"
).split()
ENGLISH = ["cricket", "computer", "internet", "mobile", "online", "PSL", "COVID-19", "team", "it's", "don't"]
WEIGHTS = [1 / (i + 1) for i in range(len(WORDS))]


def sentence(rng: random.Random) -> str:
    words = rng.choices(WORDS, WEIGHTS, k=rng.randint(4, 14))
    if rng.random() < 0.2:
        number = rng.choice([str(rng.randint(1, 99)), str(rng.randint(100, 99_999)), f"{rng.randint(1900, 2025)}ء"])
        words.insert(rng.randrange(len(words) + 1), number)
    if rng.random() < 0.15:
        words.insert(rng.randrange(len(words) + 1), rng.choice(ENGLISH))
    if rng.random() < 0.3 and len(words) > 4:
        i = rng.randrange(1, len(words) - 1)
        words[i] += "،"
    return " ".join(words) + rng.choice(["۔", "۔", "۔", "؟", "!"])


def urdu_text(n_bytes: int, seed: int = 1337) -> str:
    """About n_bytes of documents, each followed by <|endoftext|>; the same text for the same arguments."""
    rng = random.Random(seed)
    docs: list[str] = []
    size = 0
    while size < n_bytes:
        paragraphs = [" ".join(sentence(rng) for _ in range(rng.randint(1, 5))) for _ in range(rng.randint(1, 4))]
        doc = "\n".join(paragraphs) + ENDOFTEXT
        docs.append(doc)
        size += len(doc.encode("utf-8"))
    return "".join(docs)
