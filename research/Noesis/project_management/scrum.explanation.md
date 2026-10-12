# Sprints and coordination for Noesis

## Purpose and scope

Use short feedback cycles to deliver observable progress toward the Noesis MVP.
This is the team's lightweight sprint workflow, informed by Scrum; it does not
require every formal Scrum event or create additional mandatory meetings.
For the framework itself, see the [official Scrum Guide](https://scrumguides.org/scrum-guide.html).
The previous 2012-book notes have been replaced with the local workflow below.

## Sprint cadence

Each sprint is **one week, Thursday through Wednesday**. The team meets **every
Thursday to check blockers**, review the previous sprint, and coordinate the new one.
Wednesday closes the prior sprint. Meeting time, timezone, URL, and facilitator remain
to be agreed. The [project process](github_projects_process.reference.md#sprint-process)
is the source of truth for this cadence.

## Goals and backlog

- **Product goal:** the outcome the team is working toward; consult the
  [MVP requirements](../docs/mvp_prd.md) and [execution plan](../docs/mvp_plan.md).
- **Backlog:** scoped work on the Noesis board that is not yet complete. Keep
  near-term work specific and split large tasks into reviewable deliverables.
- **Sprint goal:** one clear outcome for the Thursday–Wednesday week, such as
  “Evaluate a contract automatically using saved request evidence”.
- **Sprint work:** issues selected to support that goal, with owners, acceptance
  criteria, and effort estimates in hours when useful.

Plan against actual availability; a one-week sprint does not imply weekend work.
Revisit scope when urgent work or new findings affect capacity and record the trade-off.

## Thursday blocker meeting

Before the meeting, update issues and review Wednesday's closing summary. In the meeting:

1. Briefly inspect delivered results and incomplete work from the previous sprint.
2. Review each blocker and identify the help or decision needed.
3. Give each unblock action an owner and a follow-up date.
4. Agree the new sprint goal, select issues, and explicitly decide carryover.
5. Record one process improvement if needed, along with decisions and next steps.

Discuss evidence or show a short demonstration when it helps assess progress. Arrange
separate technical follow-ups for lengthy problem-solving. This meeting is about
coordinating work, not reporting percentages complete.

## Between meetings

Use short asynchronous issue updates on days you work: what changed, what comes next,
and what blocks progress. Report blockers immediately; do not wait for Thursday.
A daily synchronous stand-up and an hour-based burndown chart are not required by this
local workflow. The team can add them if they solve a real coordination problem.

## Definition of Done

An issue is Done when its acceptance criteria are met, evidence is available, required
checks and review are complete, and any required PR is merged. Research-only work has
an accessible result and agreed review. Insufficient evidence or outstanding review
means the work remains unfinished.

The board uses only **Todo, In Progress, Blocked, and Done**. Ordinary PR review remains
In Progress. See the [GitHub guide](use_github.how_to_guide.md) for detailed transitions
and closure. Record incomplete work honestly at Wednesday's close and agree whether to
carry it forward on Thursday; preserve its history before changing the Sprint field.

## Responsibilities

The maintainer coordinates priorities and acceptance; issue owners deliver and update
work; reviewers evaluate it; the facilitator runs the weekly discussion. Confirm named
owners with the team rather than importing roles from another organization. For formal
Scrum, Product Owner, Scrum Master, and Developers have specific accountabilities
explained in the official guide; they are not new job titles assigned by this document.
