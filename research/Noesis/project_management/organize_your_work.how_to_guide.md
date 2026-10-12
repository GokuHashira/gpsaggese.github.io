# Organize your Noesis work

## Shared expectations

Be respectful, helpful, and clear about your availability. Agree on realistic weekly
commitments and communicate changes early. Contributors have different experience
and schedules; ask for help, share context, and provide constructive feedback.

Use the [Noesis board](https://github.com/users/gpsaggese/projects/2) to make work
visible. The cadence is **one-week Thursday–Wednesday sprints**, with a **Thursday
meeting to check blockers**. The [project process](github_projects_process.reference.md)
defines the board and meeting agenda.

## Responsibilities

- **Project maintainer or technical lead:** clarify priorities and interfaces, help
  unblock contributors, coordinate review, and confirm acceptance of completed work.
- **Issue owner:** keep the issue and board current, provide reproducible results,
  flag dependencies, and follow through on review feedback.
- **Reviewer:** assess the proposed change and validation, give actionable feedback,
  and communicate review delays that affect the sprint.
- **Meeting facilitator:** guide the Thursday blocker review and record decisions,
  unblock actions, owners, and follow-up dates. The team chooses the facilitator;
  this document does not assign a person.

Pairing and shared research are welcome. Keep one accountable issue owner and agree
how collaborators divide work to avoid duplicate or conflicting changes.

## Starting work

Read the [onboarding guide](../docs/onboarding.md), [MVP plan](../docs/mvp_plan.md),
and the issue's acceptance criteria. Clarify missing requirements before substantial
implementation. Break work into reviewable steps that fit available sprint capacity.
Use a branch in the repository when authorized, or a fork when repository write access
is unavailable. Use the documented environment so others can reproduce the result.

## Recording research and progress

Keep code and reviewed documentation in research/Noesis. Record experiments and
findings in docs/findings with the configuration, commands, model/provider identity,
observed results, limitations, and links to evidence. Link the record from its issue
and PR. Do not commit secrets, private data, or large raw artifacts without an agreed
storage approach.

A shared document may supplement repository notes if everyone can access it. A Noesis
shared Drive folder has not been specified; do not use an inherited KaizenFlow folder.
Keep Git checkouts outside cloud-synced folders, consistent with the onboarding guide.

On days you work, leave a short issue update: completed work, next step, and blockers.
Commit reviewable progress and use draft PRs when early feedback would help. See the
[GitHub guide](use_github.how_to_guide.md) for submission and acceptance.

## Blockers and communication

Record a blocker in the issue, move it to Blocked, and mention who can help. State
what you tried, the impact, and a concrete next action. Use the team's agreed urgent
channel for time-sensitive help, then record the outcome on the issue. The channel
and meeting URL still need confirmation.

Do not wait for Thursday to report a blocker. At the Thursday meeting, review open
blockers and agree owners and follow-up dates. Keep deep technical discussion in
focused follow-ups, and preserve decisions in the issue or weekly summary.

## Finishing and choosing next work

Show how acceptance criteria were met and link validation evidence. Address review
feedback and complete the [closing process](use_github.how_to_guide.md#closing-an-issue).
If you finish early, coordinate with the team before starting another priority; record
any resulting sprint scope change. Preserve carryover decisions at the Wednesday close.
