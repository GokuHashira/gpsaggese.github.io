# Use GitHub for Noesis

## Start here

- [Noesis project board](https://github.com/users/gpsaggese/projects/2)
- [Repository issues](https://github.com/gpsaggese/gpsaggese.github.io/issues)
- [Repository pull requests](https://github.com/gpsaggese/gpsaggese.github.io/pulls)
- [Board and sprint process](github_projects_process.reference.md)

Executable work belongs in an issue. Ideas may begin in planning notes before becoming
scoped issues; see the [issue workflow](issue_workflow.explanation.md).

## Sprints and statuses

Sprints run **Thursday through Wednesday**, one week at a time. The team meets
**every Thursday to check blockers**, review the previous sprint, and agree next
steps. See the [sprint process](github_projects_process.reference.md#sprint-process)
for preparation, carryover, and the weekly summary.

| Status | Meaning |
| --- | --- |
| Todo | Not started |
| In Progress | Active work, testing, or normal PR review |
| Blocked | A dependency or decision prevents progress |
| Done | Acceptance criteria met, review complete, and required PR merged |

PR feedback does not introduce another board status. Keep the issue In Progress
while addressing comments. Use Blocked only when an obstacle prevents useful progress;
record who can help and the next action. After it clears, return to In Progress or Todo.

## Filing an issue

1. Use a short, descriptive action title, such as “Add automatic contract evaluation”.
2. Explain the problem, expected outcome, scope, and acceptance criteria.
3. For bugs, provide reproduction steps, expected versus actual behavior, and relevant
   text logs or errors. Screenshots can supplement logs for visual problems.
4. Add related issues, design documents, and dependencies. Use checklists for small
   steps; split independently deliverable work into separate issues.
5. Add the issue to the Noesis board and choose Todo.
6. Assign one accountable owner before implementation begins. If ownership is unclear,
   ask the project maintainer; name collaborators separately.
7. Use appropriate existing [repository labels](https://github.com/gpsaggese/gpsaggese.github.io/labels).
   Do not assume imported labels or label-sync scripts are configured here.

Use priority labels only when the team agrees on their meaning. A high-priority task
is not necessarily blocked, and a Blocked status does not itself indicate priority.

## Updating an issue

Move it to In Progress when work starts. Update it on days you work with what changed,
what remains, relevant evidence, and links to PRs or findings. Limit concurrent work
so that near-complete tasks receive attention before starting more tasks.

When blocked, move it to Blocked and record:

```text
Blocker:
Impact:
Help needed from:
Next action and owner:
Follow-up date:
```

Mention the person who can resolve it promptly. The Thursday meeting follows up on
open blockers; it is not the first opportunity to request help.

If work is deferred, return it to Todo and explain why. Preserve sprint history in
comments or the weekly summary before changing its sprint assignment.

## Opening a pull request

1. Work on a feature branch, not master. Follow repository branch conventions.
2. Stage and inspect only the files intended for that PR. Keep unrelated work separate.
3. Commit and push the branch, then open a PR against master.
4. Write a descriptive title and explain the problem, change, validation, and remaining
   limitations. The title does not have to match the branch name.
5. Reference the issue using “Related to #123”. Preserve manual issue closure for this
   workflow: avoid closing keywords and automatic-closing links when more work remains.
6. Request a teammate's review. Draft PRs are useful for incomplete work or design discussion.
7. Run relevant tests and required repository checks. For documentation, check links,
   accuracy, and formatting; for websites, check layout and interactions as well.
   Report checks that could not be run.

Optional PR labels such as PR_for_authors, PR_for_reviewers, and PR_for_integrators
may be used if they exist in the repository, but GitHub review requests, drafts,
and comments remain the source of review state. No label is a substitute for approval.

## Review and feedback

Reviewers check correctness, scope, interfaces, readability, and validation evidence.
Authors address agreed feedback wherever it applies, explain disagreements, and
re-request review when ready. Resolve threads after addressing the concern according
to the reviewer's expectations; retain unresolved discussions until there is agreement.
Keep the issue In Progress until accepted and merged, unless genuinely blocked.

## Closing an issue

For code or documentation changes, required review and checks must be complete and
all required PRs merged into master. For research or coordination work without a PR,
attach the result and obtain the agreed review. A maintainer or designated reviewer
confirms acceptance, closes the issue, and sets the board status to Done.

Resolve each checklist item, or explicitly move it to a linked follow-up issue with
an agreed scope change. Author-complete is not Done while review or acceptance remains.
If closing as obsolete or duplicate, explain the reason and record that disposition
in the weekly summary; do not report it as delivered work.

## GitHub Actions

Repository workflows live under .github/workflows. Inspect the repository's actual
checks before relying on them; a green PR does not prove Noesis tests ran if the
workflow excludes this directory. Follow the [Noesis onboarding guide](../docs/onboarding.md)
for local validation. Changes to workflow behavior should themselves be reviewed.

For reference, see [GitHub issue and PR linking](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue).
