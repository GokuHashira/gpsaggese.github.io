# Workflow

## Step 1
- Read `<FILES>`
book.springer/lectures_source/Lesson01.01_From_Data_Science_To_Decision_Science.txt
book.springer/lectures_source/Lesson02.01_Integrating_Causality_And_Probability_in_ML.txt
book.springer/lectures_source/Lesson03.01_Integrating_Business_Objective_And_Real_World_Dynamics.txt

## Step 2: Collect the material
- Read the existing content in `<FILES>`
  - Print a summary of what you have learned

- Read the slide material pointed by `### Lessons`
  - Print a summary of what you have learned

## Step 3
- Execute `/Users/saggese/src/umd_classes2/helpers_root/.claude/skills/slides.write/SKILL.md`

- Create 3 lessons of each 30 slides about using the material read in Step 2

- If you are using slides from `### Lessons` or materials from `### Related
  papers`
  - Add a comment like `// From: <file>:<line num> '<source line>'` right
    after `* <slide title>` to point back to where the material is coming from
    - `<file>` is a repo-root-relative path to the exact source (a lecture
      `.smd` file, or `<book>/related_papers.md` / `<book>/related_books.md`
      for a paper/book citation), `<line num>` its line, `'<source line>'`
      that line's full text single-quoted verbatim, leading marker included
      (`* `, `## `, `- [P007] `, ...) — not just the title with the marker
      stripped
      ```
      // From: msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd:318 '* Where ML Falls Short'
      ```
  - Stack one `// From: ...` line per source when a slide draws on more than
    one, lecture sources first, then paper/book sources
    ```
    // From: msml610/lectures_source/Lesson08.1-Causal_AI_intro.smd:345 '* The Cost of Ignoring Causality (1/2)'
    // From: msml610/lectures_source/Lesson11.1-Decision_Making_with_Causal_Models.smd:161 '* Confounding: The Root Cause'
    // From: book_springer/related_papers.md:12 '- [P007] Zhang et al., "A Survey on Causal Inference", 2020'
    ```
  - If no source slide can be identified for one of the lines, leave it as a
    bare `// From:` (nothing after the colon) instead of guessing
  - See `.claude/skills/slides.rules.md` ("Source Attribution") for the full
    convention

- When generating the text read the previous already generated chapters
  `book.springer/lectures_source/Lesson.*` and make sure that
  - There is a single logic flow with the rest of the lessons
  - There is no repeatition with respect to the rest of the lessons

- Save each slides in a new file, as `FILE.new.txt`

- Do not run the linter on the slides

# Conventions

## Style rules learned from edits (apply these when generating)

- General prose and markdown conventions live in `.claude/skills/text.rules.md`
  and `.claude/skills/markdown.rules.md` follow those
  - The rules below are the slide-specific additions on top of them.

- Do not sprinkle inline `Lesson X.Y` back-references inside bullets (e.g.
  "Lesson 16.4 studied...", "(Lesson 16.2)")
  - State the concept directly instead of citing where it was first taught
  - Cross-lesson continuity belongs in the roadmap/motivation slide, not
    repeated throughout the body

- Slide/section titles: drop generic label words that duplicate the first
  bold body-label (`Problem:`, `Motivation:`, `Question:`, `Implication:`,
  `Recap:`)
  - E.g., `* Problem: Why X Struggles` $\to$ `* Why X Struggles`
  - Keep `Step N:` prefixes for ordered algorithm slides
  - Titles should be short noun phrases (2-6 words)

- One **Key idea**/**Interpretation** bullet per slide is enough; do not add
  a second bullet at the end of a slide that just restates the same point in
  different words

- Use the project's vector/matrix LaTeX macros (`\vx`, `\vp`, `\mW`, ... per
  `slides.rules.md`) instead of `\mathbf{...}` for vectors and matrices in
  equations

- Do not spin up an extra slide (or table) purely to restate a distinction
  already conveyed by nearby bullets; fold it inline instead

## Structural / formatting rules learned from edits (apply these too)

- File header: use the metadata-comment block, not a LaTeX title slide
  - Start the file with, in order:
    `// type=UMD_slides`,
    `// course_title=<course>`,
    `// lesson_title=<Lesson X.Y: Title>`,
    `// slides_engine=typst`
  - Then a blank line and `// References:` with each reference as a `//-`
    comment line (URL wrapped onto a `//  ` continuation line)
  - Do NOT emit a `/* ::: columns ... */` LaTeX title slide, `\vspace`,
    `\begingroup`, instructor line, or a `**References**` bullet block

- Tables: render as a Typst `#styled-table` block, never as a markdown pipe
  table wrapped in `\begingroup \scriptsize ... \endgroup`
  ````
  ```{=typst}
  #styled-table(
    headers: ("A", "B"),
    rows: (("x", "y"),
    ),
    )
  ```
  ````
  - Order rows in the natural reading order (e.g. cheap$\to$expensive,
    zero-shot$\to$few-shot), not the order they happened to be written

- When a slide pairs prose with a figure (graphviz/diagram) or a table, put
  them side by side in columns rather than stacking:
  `::: columns` / `:::: {.column width=50%}` (text) / `:::: {.column
  width=45%}` (figure) / close with `::::` and `:::`
  - Keep the `**Key idea**` bullet in the text column, not under the figure

- Do not wrap `##` section headers in decorative comment banners
  (`## ###########...`); a bare `## Section Title` is enough

- Do not introduce a parenthetical abbreviation you will not reuse; spell out
  the term instead (removed `(open IE)`, `(PPR)`, `(BPE)`, `(nucleus)`)
  - Conversely, expand an acronym on first use when it aids reading, e.g.
    `MLP` $\to$ `two-layer MLP (multi-layer perceptron)`

- Refinement of the no-inline-back-reference rule: strip inline parentheticals
  like `(Lesson 16.2)` and forward-looking asides like
  `(previewed for later lessons)` / `revisited in the ... lesson`, BUT a
  single dedicated `**Remark**: <topic> is covered in detail in Lesson X.Y`
  is acceptable to defer a full treatment to its own lesson

- Phrase `**Question**` bullets as actual questions ending in `?`

- Prefer compact multiplicative notation over verbal forms:
  `2x the context length $\implies$ 4x the attention cost`, not
  `doubling the context length _quadruples_ the attention cost`

