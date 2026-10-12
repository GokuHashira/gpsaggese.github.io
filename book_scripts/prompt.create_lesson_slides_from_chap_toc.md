- You are a college professor expert in AI, machine learning, and big data

# Goal
- The relevant variables are
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="book_AI_Software_Engineering"
  CHAP="## 14: Human-AI Collaboration Patterns"
  ```

- The format of a book map is described here
  `helpers_root/.claude/skills/book.rules.md`

- Read carefully the entire table of content `$DIR/$BOOK/map.md`

## Write Slides
- Execute `$DIR/helpers_root/.claude/skills/slides.write/SKILL.md` for the
  material in `$CHAP` from `$DIR/$BOOK/map.md`

- Read the referred papers from the section

- Create 30 slides about 
  - When generating the text read the previous already generated chapters
    `$DIR/$BOOK/lectures_slides/Lesson*` and make sure that there is no
    repeatition (you can have lectures pointing back to others) and a single
    logic flow
- Save the slides in a new file `$DIR/$BOOK/lectures_sources/Lesson<CHAP_NUM>.<Title>.txt`

## Style rules

- Follow the general prose and markdown conventions in `.claude/skills/text.rules.md`
  and `.claude/skills/markdown.rules.md`
- Follow the slide-specific conventions in `.claude/skills/slides.rules.md`
  (cross-referencing, titles, tags, table row order, LaTeX macros, columns
  layout, ...)

## Ask questions
- If you have doubts, do not execute the task but add questions

<!-- ~/src/umd_classes1/book_AI_Software_Engineering/map.md -->
