# Noesis GitHub Projects process

## Overview

Track Noesis work in the [Noesis GitHub Project](https://github.com/users/gpsaggese/projects/2).
Issues and pull requests live in the [repository](https://github.com/gpsaggese/gpsaggese.github.io).
This document defines the board and sprint cadence. See the [GitHub guide](use_github.how_to_guide.md)
for issue and PR steps and the [team guide](organize_your_work.how_to_guide.md) for collaboration.

## Board structure

The board is grouped by Status into four columns:

| Status | Meaning | When to move the issue |
| --- | --- | --- |
| Todo | Work has not started | Add a scoped issue with acceptance criteria |
| In Progress | Work is active, including implementation, testing, or PR review | An owner starts work; keep it here while awaiting normal review |
| Blocked | Work cannot proceed because of a dependency or unresolved decision | Record the blocker, who can resolve it, and the next action |
| Done | Work meets its acceptance criteria and has been accepted | Required checks and review are complete, and any required PR is merged |

A card represents an issue and can show an assignee and linked PR. Review is tracked
through the PR; there is no separate In Review, QA, or Paused board column.
When a blocker clears, return the issue to In Progress, or Todo if work has not started.
Work deliberately deferred goes to Todo with an explanation; do not call it blocked
unless a dependency actually prevents progress.

## Issue fields

- **Assignee:** one accountable owner, with collaborators named in the issue.
- **Status:** one of the four board statuses above.
- **Labels:** use relevant labels available in the [repository label list](https://github.com/gpsaggese/gpsaggese.github.io/labels).
  Priority and work type are separate from board status.
- **Estimate:** approximate uninterrupted effort in hours when helpful. An estimate
  is a planning aid, not a guarantee; account for each person's available time.
- **Sprint:** the Thursday–Wednesday week selected for the issue.
- **Pull request:** reference the PR and related issue in both directions.

Assignee and Status are required before work starts. Sprint and Estimate are planning
fields; their existence on the live board still needs confirmation. If Sprint is not
configured, record the dates in the weekly summary until it is available.

## Sprint process

Sprints last **one week, from Thursday through Wednesday, inclusive**. The next sprint
begins on the following Thursday. Example: October 15–21, 2026, followed by October 22–28.

The team meets **every Thursday to check blockers** and coordinate the new sprint.
The meeting time, timezone, meeting URL, and facilitator remain to be agreed.

### Before the Thursday meeting

- By Wednesday's sprint close, update assigned issues with progress, evidence,
  remaining work, blockers, and any PR awaiting review.
- Review completed work against the sprint goal. Record unfinished items as
  unfinished, even if most of the implementation is complete.
- Prepare a brief weekly summary in the project status update, or in a linked
  issue if that feature is unavailable to the team.

### Thursday meeting agenda

1. Review the previous Thursday–Wednesday sprint's outcomes and unfinished work.
2. Review each Blocked issue: cause, impact, help needed, and next action.
3. Give each unblock action an owner and a follow-up date.
4. Confirm the new sprint goal, issue owners, and realistic capacity. Re-plan carryover
   explicitly; it is not automatically committed to the next sprint.
5. Record decisions, links, and next actions in the weekly summary.

Raise blockers immediately in the issue and notify the person who can help. Do not
wait until Thursday if the team can resolve the blocker sooner. Keep extended technical
problem-solving in a separate discussion with the necessary people.

### During the sprint

Update the board as work changes and leave a concise issue update on days you work.
New urgent work may enter the sprint; record what was added and what was displaced.
Before moving an unfinished issue to a later sprint, preserve its previous dates,
outcome, and reason for carryover in the weekly summary or an issue comment. A single
Sprint field is not the historical record of every week spent on that issue.

Suggested weekly summary:

```text
Sprint: Thursday YYYY-MM-DD – Wednesday YYYY-MM-DD
Goal:
Completed: issue / merged PR / evidence
Unfinished: issue / remaining work / carryover decision
Blockers: issue / cause / unblock owner / next action / follow-up date
Next sprint: goal / issues / owners
Thursday meeting decisions:
```

## Automation

Board updates are manual unless a workflow has been verified on the Noesis project.
Do not assume automation from other organizations exists here.

An authorized project maintainer can configure a Sprint iteration starting on a
Thursday with a one-week duration; see [GitHub iteration fields](https://docs.github.com/en/issues/planning-and-tracking-with-projects/understanding-fields/about-iteration-fields).
Do not automatically roll over unfinished issues without recording the sprint outcome.
Built-in workflows can reduce manual updates; see [GitHub project automations](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-built-in-automations).
Any automation must preserve the Done criteria above.

## Larger work

Break work that cannot reasonably fit in a sprint into independently reviewable
issues. The existing convention is to avoid GitHub sub-issues; reference related
issues and use checklists for small steps. Revisit this convention explicitly if
hierarchical tracking becomes useful.
