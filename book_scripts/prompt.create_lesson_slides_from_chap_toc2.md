# Goal
- Create lectures slides of a book from chapter TOC

# Workflow
- Information about the entire book flow is at `book_proposals/README.md`

- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_AI_Software_Engineering"
  CHAP="## 03, Prompt Engineering for Code"
  ```

## Step 1: Understand the Plan
- Read carefully the entire table of content `<DIR>/<BOOK>/map.md` of the book

- Read the target chapter `<CHAP>` that you need to convert from TOC into slides

## Step 2: Collect the Related Material
- Read the topics in `### Topics` from the book map `<DIR>/<BOOK>/map.md`
- Read
  - The slide material pointed by `### Lesson Materials` if present
  - The pointers in `<DIR>/<BOOK>/resources.md`

## Step 3: Create Slides
- Execute `<DIR>/helpers_root/.claude/skills/slides.write/SKILL.md`

- Create 30-35 slides about the Topics in `### Topics` using the material
  read in Step 2

- If you are using slides from `### Lesson Materials` or materials from
  `### Related papers`, add a comment like `// From: <file>:<line num>
  '<source line>'` right after `* <slide title>` to point back to where the
  material is coming from
  - `<file>` is a repo-root-relative path to the exact source (a lecture
    `.smd` file, or `<book>/related_papers.md` / `<book>/related_books.md`
    for a paper/book citation), `<line num>` its line, `'<source line>'` that
    line's full text single-quoted verbatim, leading marker included (`* `,
    `## `, `- [P007] `, ...) — not just the title with the marker stripped
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
  `<DIR>/<BOOK>/lectures_source/Lesson*` and make sure that there is:
  - A single logic flow with the rest of the lessons
  - No repeatition with respect to the rest of the lessons

## Step 4: Save Results
- Save the slides in a new file in `<DIR>/<BOOK>/lectures_source` based on the
  name of the target H2 chapter
  - E.g., 
    `## 01: Why Decisions, Not Predictions` ->
    `book.springer/lectures_source/Lesson01.01_Why_Decisions_Not_Predictions.txt`
  - If the file already exist, just overwrites it

# Conventions
- Follow:
  - The workflows in
    `/Users/saggese/src/umd_classes2/helpers_root/.claude/skills/slides.write/SKILL.md`
  - All formatting rules, templates, and structural guidelines in
    `.claude/skills/slides.rules.md`

// Ignore
// From /Users/saggese/src/umd_classes1/book_AI_Software_Engineering/map.md
// BOOK="book_AI_Software_Engineering"
// CHAP="## 01: LLM-Assisted Code Generation"
// CHAP="## 02: AI Pair Programming Workflows"
// CHAP="## 03, Prompt Engineering for Code"
// CHAP="## 04, Retrieval-Augmented Generation for Codebases"

// CHAP="## 14: Human-AI Collaboration Patterns"
//
// BOOK="book_Decentralized_AI"
// CHAP="## 01: Blockchain Consensus Foundations"
