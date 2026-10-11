"""Urdu text normalization: normalize(text) -> str.

Applied while cleaning the corpus (clean/ pipeline), never inside the tokenizer.
Covers the handout's table (docs/assignment.md, "Urdu normalization"): Arabic
yeh/alef maksura -> Urdu yeh, Arabic kaf -> Urdu kaf, Arabic heh -> gol heh
(do-chashmi heh kept apart), tatweel, invisible characters (ZWNJ policy
decided and justified in docs/decisions.md), one digit convention, and
whitespace collapse. Every line must be our own work; no Urdu NLP library may
do this for us.

Not implemented yet (Step 0 scaffold).
"""
