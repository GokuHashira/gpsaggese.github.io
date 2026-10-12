# Noesis issue workflow

```text
Idea → scoped project or milestone → actionable issue → reviewed result
```

## Idea

An idea describes an opportunity before its implementation is clear. For example:
“Can we detect when a provider serves a weaker model than promised?”

Record the problem, intended benefit, and unknowns in a shared planning note or a
clearly marked exploratory issue. Link it from relevant work so the team can find it.
A Google Doc may be used if shared with the team; it is not a required separate system.

## Project

A project has a goal, scope, intended outcome, and constraints. Noesis already has
[a project board](https://github.com/users/gpsaggese/projects/2); individual milestones
and features do not each require a new board.

Before implementation, the team identifies the problem, considers alternatives,
splits the work into issues, identifies dependencies, and estimates effort. Use the
[MVP plan](../docs/mvp_plan.md) and [requirements](../docs/mvp_prd.md) to connect tasks
to the intended product outcome. Record decisions that change scope or interfaces.

Example: the verified-delivery milestone becomes separate issues for calibration,
probe scheduling, automatic evaluation, and a reproducible fault experiment.

## Task (issue)

An issue is a specific piece of work with one accountable owner and a testable result.
Include:

- Problem and why it matters.
- Scope and explicit exclusions.
- Acceptance criteria: what evidence will show the task is complete.
- Dependencies, owner, and an effort estimate when helpful.
- Relevant design notes, PRs, and findings.

Add executable Noesis issues to the project board. Use [Todo, In Progress, Blocked,
and Done](github_projects_process.reference.md#board-structure) consistently.
Select work for **one-week Thursday–Wednesday sprints**, and review blockers at the
**Thursday meeting**. Report blockers as they occur.

For example, “Run provider calibration” is complete when the selected configuration,
results, costs, and retained question set are documented and reviewed—not merely when
the script runs.

See the [GitHub guide](use_github.how_to_guide.md) for filing, reviewing, and closing work.
