# Step 12 hand-off: what runs on the corpus machine

Everything that needs the corpus (`data/`) or TeX runs on the machine that holds `data/final/`. The code, the report source, the data card and the README are finished. Run these steps in order from `~/Desktop/LLM/urdu-bpe`. Each step says what it should print. If a step prints something else, stop and send the output.

1. **Get the code.**
   ```bash
   git pull
   ```
   If git refuses because of local changes (e.g. an edited `docs/analysis_notes.md`), run `git stash && git pull && git stash pop`.

2. **The shared-drive link.** Put the Drive folder link (the one Syed has) on one line in `drive_link.local`:
   ```bash
   printf '%s\n' 'PASTE-THE-FOLDER-LINK' > drive_link.local
   ```
   The file is gitignored. The repository is public, so the link goes only into the README inside the submission zip.

3. **TeX tools and fonts.**
   ```bash
   .venv/bin/python report/checks.py tools
   ```
   It must show paths for lualatex and latexmk and resolve "Noto Naskh Arabic". If anything is missing, install BasicTeX (about 100 MB, much smaller than MacTeX), then open a new terminal:
   ```bash
   brew install --cask basictex              # asks for the admin password
   sudo tlmgr update --self
   sudo tlmgr install latexmk fontspec luaotfload babel-english booktabs geometry hyperref xcolor
   brew install --cask font-noto-naskh-arabic
   brew install poppler                      # pdfinfo; optional
   luaotfload-tool --update
   ```

4. **Tests.**
   ```bash
   .venv/bin/python -m pytest -q
   ```
   Expect "N passed" with no failures.

5. **Fast vs reference trainer, as JSON for the report.**
   ```bash
   .venv/bin/python analysis/equivalence_1mb.py
   ```
   Expect "Result: IDENTICAL". It writes `data/stats/equivalence_1mb.json` and refreshes `docs/equivalence_1mb.md`.

6. **Part C analysis.**
   ```bash
   make analysis
   ```
   This rewrites the tables, the figures and the generated blocks of `docs/analysis_notes.md`; text outside the BEGIN/END markers is kept. Q1 now says 21 two-byte letters plus 2 other two-byte characters (۔ and ،).

7. **Data card.**
   ```bash
   .venv/bin/python clean/data_card.py
   git diff --stat DATA_CARD.md
   ```
   Expect no change: the committed card already has the final gaps section, without REVIEW markers. If it does change, keep the new version.

8. **The report's numbers.**
   ```bash
   make numbers
   ```
   It writes `report/numbers.tex` and `report/tables/r_*.tex`. If it stops with "missing inputs" or a missing key, send the message.

9. **Build the report.**
   ```bash
   make report
   ```
   Expect "0 missing glyph(s)" and "report.pdf: N pages ...; limit 4: OK".
   - **Over 4 pages:** in `report/report.tex`, change `\newcommand{\tablefont}{\footnotesize}` to `\scriptsize` and run `make report` again.
   - **A LaTeX error or missing glyphs:** send `report/report.log`.

10. **Pre-flight and packaging.**
    ```bash
    make submission
    ```
    Every line must be [PASS]. It writes `submission/urdu_bpe_456094.zip` and `data/final/SHA256SUMS`, and prints the files to upload with their sizes.

11. **Upload to the Drive folder.**
    - **Files:** `data/final/train.txt`, `val.txt`, `test.txt`, `train.bin`, `val.bin`, `meta.json`, `manifest.json` and `SHA256SUMS`.
    - **Sharing:** the folder must be shared with course staff only, not "anyone with the link". Makhzan was approved only on that condition.

12. **Commit and push** (no data: `data/`, `submission/` and `drive_link.local` are gitignored).
    ```bash
    git add report/numbers.tex report/tables report/figures report/report.pdf docs/analysis_notes.md \
        docs/equivalence_1mb.md DATA_CARD.md
    git commit -m "Step 12: report + packaging"
    git push
    ```

13. **Submit** `submission/urdu_bpe_456094.zip`. The Drive link is inside its README.
