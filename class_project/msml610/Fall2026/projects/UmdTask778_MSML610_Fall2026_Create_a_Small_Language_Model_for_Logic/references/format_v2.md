# Output Format v2 (2026-11-18)

v2 extends [v1](format_v1.md) for real (human / LLM written) math datasets.
`FORMAT_VERSION = "v2"` in `logic_slm_format.py`, which stays the single
reference implementation. **Everything v1 renders is rendered byte-identically
in v2**; synthetic generators (M1-M3, L*, TinyGSM) are unchanged.

## 1. Layout (unchanged)

```
Problem: <question>
Solution:
<line 1>
...
<line k>
#### <answer>
```

- The prompt is `Problem: <question>\nSolution:\n` (math) or
  `Premises: ...\nQuestion: ...?\nSolution:\n` (logic).
- The solution is zero or more lines, then exactly one `#### <answer>` line,
  with no trailing newline (the tokenizer appends EOS).
- Parsing (`parse_answer`, `parse_steps`) is unchanged.

## 2. Real-data solutions

- Solution lines are **free text** (sentences, LaTeX allowed), one or more
  sentences per line, as the source wrote them, stripped of blank lines.
  They are not checked by `check_math_step` (v1 math-step rule applies only to
  synthetic data).
- No line may start with `####`; the final answer line is always produced by
  `render_solution` from the cleaned answer.
- Source-specific cleaning (see `logic_slm_realdata.py`):
  - GSM8K: calculator annotations `<<48/2=24>>` are removed; answer after
    `####`.
  - OpenMathInstruct-2: the final sentence containing `\boxed{...}` is
    removed; the answer is `expected_answer`.
  - MetaMathQA: the trailing `The answer is: X` line is removed; answer `X`.
  - Orca-Math: the final answer is extracted from the last sentence(s)
    (`is 42.`, `\boxed{42}`, `= 42`); examples without a clean numeric final
    answer are dropped.
- The real-data question may span several lines in the source; newlines in
  the question are collapsed to spaces so the prompt is always one
  `Problem:` line.
- Records carry `gen` (source name) and `level` (= source) like synthetic
  records, plus source metadata (e.g. `problem_source`).

## 3. Answers

Gold answers are stored as the source gives them after light cleaning (e.g.
`18`, `3.5`, `\frac{1}{2}`, `[2,5)`). Comparison uses
`normalize_answer(answer, "math")`:

1. Collapse whitespace.
2. Remove LaTeX wrappers: last `\boxed{...}` / `\fbox{...}` (content kept),
   outer `$...$`, `$$...$$`, `\(...\)`, `\[...\]`; `\dfrac`/`\tfrac` ->
   `\frac`; `\frac12` -> `\frac{1}{2}`; `\left`, `\right`; `\text{..}`,
   `\mathrm{..}`, `\mbox{..}` (content kept); LaTeX spacing (`\,` `\!` `\;`
   `~` `\quad`); `^\circ`; `\%` -> `%`, `\$` -> `$`; unicode minus -> `-`.
3. **Numeric answers** compare by exact value and are rendered canonically as
   an integer or reduced fraction (`7/2`, `-3/4`). Accepted: integers,
   decimals (`3.50`, `.5`), thousands separators (`1,234.5`), `a/b`,
   `\frac{a}{b}`, a leading `$` (`$5`, `-$5`, `$-5`), a trailing `%`, a
   leading `x =`, a trailing `.`, and at most two trailing unit words from a
   fixed whitelist (dollars, cents, percent, degrees, time units, length /
   mass / volume units, `square`, `units`, ...). Object nouns are never
   stripped (`12 apples` stays non-numeric, as in v1). So
   `3.50 == 3.5 == 7/2 == \frac{7}{2} == \boxed{3.5} == $3.50`.
4. **Non-numeric answers** (MATH-style intervals, expressions, words) keep the
   cleaned string from step 2 with spaces next to punctuation removed:
   `\left( 3, \frac{\pi}{2} \right)` -> `(3,\frac{\pi}{2})`. Spaces between
   two word characters are kept (`12 apples`). No symbolic equivalence is
   attempted (`\frac{\sqrt{3}}{2}` != `\frac{1}{2}\sqrt{3}`).
5. A zero denominator (`1/0`) is kept as is (v1 behaviour).
6. Logic verdicts are unchanged (lowercase, trailing `.` removed).

Percent caveat: `25%` and `25` are equal (the `%` is dropped, not divided by
100), matching how GSM8K-style datasets write percentage answers.

## 4. Records (JSONL)

Training records (gzip JSONL shards):

```json
{"format": "v2", "task": "math", "prompt": "Problem: ...\nSolution:\n",
 "solution": "...\n#### 18", "answer": "18", "level": "gsm8k-train",
 "gen": "gsm8k"}
```

Evaluation records use the same schema; `solution` may be `""` when the gold
solution is not in the target format (MATH-500, SVAMP, ProofWriter).
ProofWriter items are `task="logic"` with natural-language premises and
`yes` / `no` / `unknown` answers.

## 5. Decontamination

Training shards are filtered with `logic_slm_decontam.py`: a training record
is dropped if any word 13-gram (lowercased, punctuation stripped) of its
prompt + solution occurs in an evaluation prompt.
