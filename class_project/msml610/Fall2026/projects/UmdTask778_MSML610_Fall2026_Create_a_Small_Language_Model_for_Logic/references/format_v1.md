# Output Format v1 (frozen 2026-11-15)

Every training, SFT and evaluation example uses this format. Generators,
scorers, the tokenizer and SFT loss masking all depend on it. Changes require a
new version (`format_v2.md`) and a bump of `FORMAT_VERSION` in
`logic_slm_format.py`, which is the single reference implementation.

## 1. Layout

An example is a **prompt** followed by a **solution**.

- The prompt ends with the line `Solution:` and a newline.
- The solution is zero or more step lines, then exactly one answer line
  `#### <answer>`.
- SFT computes the loss only on the solution (everything after `Solution:\n`).
- At inference the model is stopped at the end of the answer line (EOS).

### Math

```
Problem: (17 + 4) * 3 - 12 = ?
Solution:
17 + 4 = 21
21 * 3 = 63
63 - 12 = 51
#### 51
```

### Logic

```
Premises: p -> q. q -> r. p.
Question: r?
Solution:
p, p -> q => q
q, q -> r => r
#### yes
```

## 2. Math Rules

- One step per line, each step computes exactly one operation:
  `<a> <op> <b> = <c>` with `op` in `+ - * /`.
- Word problems (M3, TinyGSM) may prefix a step with a short description:
  `cost of trays: 10 * 50 = 500`. Only the part after the last `:` is checked.
- Numbers: M1 uses non-negative integers only. Later levels may use negative
  integers (`-7`) and exact fractions (`3/4`). No decimals, no thousands
  separators, no leading zeros.
- Answer canonical form: integer `51`, `-7`; fraction in lowest terms `3/4`,
  `-3/4`. Scorers normalize before comparing (`051` == `51`, `6/8` == `3/4`).

## 3. Logic Rules

- Symbols: atoms are lowercase identifiers (`p`, `q`, `rain`); connectives
  `~` (not), `&` (and), `|` (or), `->` (implies). First-order (L2):
  `forall x (A(x) -> B(x))`, `exists x (...)`, predicates capitalized,
  constants lowercase.
- Premises are separated by `. ` and the premise line ends with `.`.
- The question asks about one formula: `Question: <formula>?`.
- **Verdicts:** `yes` (the formula follows from the premises), `no` (its
  negation follows), `unknown` (neither follows). Datasets are balanced to
  about one third each.
- **Steps (forward chaining):** `<f1>, <f2>, ... => <g>` where `<g>` follows
  from the listed formulas (premises or earlier conclusions) by one inference
  rule. Every step is checked independently with z3.
  - `yes`: steps end by deriving the question formula.
  - `no`: steps end by deriving `~<question formula>`.
  - `unknown`: steps derive every fact reachable from the premises (closure),
    none of which is the formula or its negation. May be empty.

## 4. Parsing Contract (`logic_slm_format.py`)

- `parse_answer(text)`: the answer is the text after the **first** line that
  starts with `####` following `Solution:` (or anywhere, if `Solution:` is
  absent, e.g. a bare completion). Returns `None` if missing or empty.
- `parse_steps(text)`: the non-empty lines between `Solution:` and the answer
  line.
- `normalize_answer(answer, task)`: canonical form used for exact match.
