# Goal
- Create lectures slides of a book from chapter TOC

# Workflow

- The relevant variables are:
  ```
  DIR="/Users/saggese/src/umd_classes1"
  BOOK="msml610"
  ```

## Read
- Read `.claude/skills/slides.rules.md` to understand how slides are organized

## Topics
- defining a search problem via states, initial state, actions, transition model, goal test, and path cost; distinguishing the state space from the search tree/graph.
- uninformed search
- informed search
- A* search (and its property)
- Adversarial search and game trees
- minimax
- alphabeta pruning
- depth limited search
- Use tic-tac-toe and connect 4 as examples

- Align the content with
  `msml610/lectures_source/Lesson09.8-MonteCarlo_Tree_Search.smd`

## Avoid Redundancy with Existing Material
- Make sure there is no redundancy with the existing material in
  `ls msml610/lectures_source/*.smd`
  especially `msml610/lectures_source/Lesson09.8-MonteCarlo_Tree_Search.smd`

## Generate Slides
- Execute `.claude/skills/slides.write/SKILL.md` to create 30-35
  slides about the content in `## Topics`
- The output is `<DIR>/<BOOK>/lectures_source/Lesson09.9-Classical_search_algorithms.smd`

## Create a Skeleton of Slides
- Initially create a list of slides with only titles
- Have the user approve
- Then complete the slides following `.claude/skills/slides.rules.md` and
  `.claude/skills/slides.write/SKILL.md`
- If you have doubts, something is not clear, stop and ask questions before
  proceeding
