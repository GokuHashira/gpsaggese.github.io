# Organizing Software Development for Agentic AI

_Recommendations for a Python team using Claude Code_

## The Problem

AI made writing code cheap. Understanding and verifying code still costs what it
always did. When generation outpaces comprehension, you get:
- PRs too large to review properly
- AI-generated code approved by AI reviews, with no human who really understands it
- Architecture that erodes one "reasonable-looking" change at a time
- A system nobody can explain
**Principle:** treat human understanding as the scarce resource, and organize the
workflow around protecting it. Humans own intent, architecture and approval. Agents
own implementation inside clear, machine-checked boundaries

## 1. Stop the Bleeding (this Week)

### 1.1 Cap PR Size

- Hard limit of roughly **400 changed lines** per PR, excluding lockfiles, migrations
  and generated files. Enforce it in CI
- Bigger work ships as **stacked PRs**, one concern each
- A tech lead can override with a label (`large-pr-approved`) for genuine exceptions
  such as mechanical renames

### 1.2 A Human Owns and Understands Every PR

- The PR author must be able to walk a reviewer through the change **without the
  AI**. "Claude wrote it" is not an explanation. If they can't explain it, it doesn't
  merge
- The author writes the "what and why" in the PR description themselves

### 1.3 AI Review Is a First Pass, Never an Approval

- Use Claude to catch bugs, missing tests and style issues before human review
- Require at least one **human approval from the code owner** of every touched module
  (branch protection: "Require review from Code Owners")
- The same tool or person must not both generate and approve a change

### 1.4 Limit Work in Progress

- For the next few weeks, cap open PRs per engineer (e.g. 2) and merge on a steady
  cadence. A temporary slowdown is far cheaper than months of untangling a system
  nobody understands

## 2. Recover the Architecture (weeks 2–4)

Agents can't respect an architecture that exists only in people's heads, or nowhere

### 2.1 Map What Exists

- Ask Claude Code to map the repo's modules, dependencies and main data flows into
  `docs/architecture.md`, including a Mermaid diagram
- **Have a senior engineer verify every line.** An unverified AI-generated map is
  worse than none

### 2.2 Decide the Target Structure

A layered structure works well for most Python services:
| Layer          | Package                | Owns                                   | May import                         |
| -------------- | ---------------------- | -------------------------------------- | ---------------------------------- |
| API            | `myapp.api`            | HTTP routes, request/response schemas  | application                        |
| Application    | `myapp.application`    | use cases, orchestration, transactions | domain                             |
| Domain         | `myapp.domain`         | business rules, entities, pure logic   | stdlib only                        |
| Infrastructure | `myapp.infrastructure` | DB, queues, external APIs              | domain (implements its interfaces) |
Feature modules (`billing`, `accounts`, `pricing`...) must not import each other's
internals; they talk only through the public interface in `<module>/__init__.py`

### 2.3 Record Decisions as ADRs

Keep short Architecture Decision Records in `docs/adr/`. Write each decision so an
agent can follow it literally:
> All external HTTP calls go through `myapp.infrastructure.http.Client`; nothing else
> imports `httpx`

### 2.4 Assign Owners

Use `.github/CODEOWNERS` so every module has a human who understands it and must
approve changes:

```text
*                          @your-org/tech-leads
/CLAUDE.md                 @your-org/tech-leads
/.claude/                  @your-org/tech-leads
/docs/adr/                 @your-org/tech-leads
/src/myapp/**/billing/     @your-org/billing-team
/src/myapp/**/accounts/    @your-org/accounts-team
/src/myapp/api/            @your-org/tech-leads
/migrations/               @your-org/tech-leads
```

## 3. Make the Architecture Machine-enforced

Agents follow rules reliably when the rules are explicit **and** checked
automatically

### 3.1 Architecture Tests with Import-linter

Add to `pyproject.toml` and run `lint-imports` in CI:

```toml
[tool.importlinter]
root_package = "myapp"
include_external_packages = true

[[tool.importlinter.contracts]]
name = "Layered architecture"
type = "layers"
layers = [
    "myapp.api",
    "myapp.application | myapp.infrastructure",
    "myapp.domain",
]

[[tool.importlinter.contracts]]
name = "Domain has no framework or IO dependencies"
type = "forbidden"
source_modules = ["myapp.domain"]
forbidden_modules = ["fastapi", "sqlalchemy", "requests", "httpx", "boto3"]

[[tool.importlinter.contracts]]
name = "Feature modules are independent"
type = "independence"
modules = ["myapp.domain.billing", "myapp.domain.accounts", "myapp.domain.pricing"]
```

Your current code will probably fail. List today's violations under `ignore_imports`,
burn them down over time, and let any **new** violation fail CI immediately

### 3.2 Other Automated Gates

- **ruff** with complexity caps (`max-complexity = 10`, `max-args = 6`). This pushes
  agents toward small functions
- **mypy --strict** on the package
- **pytest + diff-cover**: at least 80% coverage on changed lines
- Contract tests on module public interfaces and API schemas

## 4. Configure Claude Code for the Team

Commit the `.claude/` folder and `CLAUDE.md` files to the repo so every engineer's
agent works under the same rules

### 4.1 Root `CLAUDE.md`

Loaded into every session. Keep it short and current. It should contain:
- What the system does (3–5 sentences)
- The layer table and module rules above, plus "if a change seems to need a boundary
  violation, **stop and ask**"
- The workflow: plan first, under 300 lines per change, tests first, never change a
  public interface without saying so
- Commands: install, test, lint, type check, `lint-imports`
- Don'ts: new dependencies without approval, editing merged migrations, silencing
  type or lint errors without a reason

### 4.2 Per-module `CLAUDE.md`

One per significant module, written by its owner: purpose, public interface,
**invariants that must never break**, and gotchas. Example:

```markdown
# billing
Owner: @billing-team. Invoices, payments, refunds. NOT pricing.
Public interface: create_invoice, apply_payment, refund.
Invariants:
- Invoice total is immutable once status is `issued`.
- Money is Decimal in minor units. Never float.
- Every state change writes an InvoiceEvent in the same transaction.
Gotcha: provider webhooks can arrive twice. Handlers must be idempotent.
```

### 4.3 Hooks: Catch Problems While the Agent Works, Not in CI

In `.claude/settings.json`:

```json
{
  "permissions": {
    "deny": ["Edit(migrations/versions/**)", "Edit(vendor/**)", "Bash(git push --force*)"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write",
        "hooks": [{ "type": "command",
                    "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check_after_edit.py\"" }]
      }
    ],
    "Stop": [
      {
        "hooks": [{ "type": "command",
                    "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check_diff_size.py\"" }]
      }
    ]
  }
}
```

- **After each edit:** run `ruff` on the file and `lint-imports`. On failure, exit
  with code 2 so the error is fed back to Claude and fixed immediately
- **When Claude finishes:** count changed lines against `origin/main`. Over 300,
  block and tell Claude to stop adding code and propose a split into stacked PRs

### 4.4 Custom Commands

- **`/plan <task>`** (`.claude/commands/plan.md`): Claude reads the relevant
  `CLAUDE.md` files and architecture doc, writes a plan to `docs/plans/<slug>.md`,
  and **stops without writing code**. Plan sections: Problem, Approach (with rejected
  alternative), Modules touched, Interface changes, New dependencies, Test plan, PR
  breakdown, Open questions
- **`/explain-pr`** (`.claude/commands/explain-pr.md`): produces a _reviewer's
  guide_, not a summary. It covers intent, the 3–5 places that need careful reading
  (file:line), architecture impact, risks and untested paths, and questions for the
  author. Instruct it to be skeptical and never say "looks good"

## 5. Change the Workflow: Spec First, Code Second

1. **Plan:** for anything beyond a small fix, run `/plan`. A human reviews the plan,
   which is about 30 lines instead of 2,000 lines of diff
2. **Tests:** humans write or carefully review the acceptance tests. If the tests
   correctly capture the intended behavior, the implementation matters much less
3. **Implement:** Claude implements within the plan, with hooks enforcing boundaries
   and size
4. **Explain:** the author runs `/explain-pr`, edits the output, and puts it in the
   PR description
5. **Review:** the human code owner reviews, starting with the places the guide
   flags. Public interface changes also need a tech lead
Enforce it in CI: PRs adding more than ~100 lines must link a plan (`docs/plans/...`)
in the description

### PR Template (`.github/pull_request_template.md`)

```markdown
## What and why
<!-- Written by you, not pasted from the agent. -->
Plan: docs/plans/xxx.md

## Where reviewers should look
<!-- Edited /explain-pr output -->

## Author checklist
- [ ] I can explain every part of this change without the AI
- [ ] I ran it and checked the behavior myself, not just the tests
- [ ] I reviewed the tests, not only the code
- [ ] No new cross-module dependencies or public interface changes (or listed above)
- [ ] Docs / module CLAUDE.md updated if behavior or boundaries changed
```

## 6. Rebuild Shared Understanding as a Habit

- **Weekly 30-minute architecture review:** new ADRs, structural changes, the
  `ignore_imports` count, the largest PRs, and reverts
- **Rotating walkthroughs:** an engineer presents a module they _didn't_ write. Gaps
  in understanding show up quickly
- **Docs live next to code:** a PR that changes behavior or boundaries updates
  `docs/architecture.md` or the module's `CLAUDE.md` in the same PR

## 7. Measure the Right Things

Stop celebrating output volume (PR count, lines written). Track:
| Metric                                           | Healthy direction             |
| ------------------------------------------------ | ----------------------------- |
| Median PR size                                   | Down, ideally under 300 lines |
| Time to first human review                       | Down                          |
| Revert / rollback rate                           | Down                          |
| Incidents traced to recent changes               | Down                          |
| Architecture violations (`ignore_imports` count) | Down                          |
| % of PRs with an approved plan                   | Up                            |
If PR count rises while reverts and incidents also rise, you're not actually moving
faster

## Rollout Plan

| When          | Actions                                                                                                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Week 1**    | PR size check in CI; CODEOWNERS; PR template; branch protection requiring human code-owner approval; WIP limits                                                                             |
| **Weeks 2–4** | Architecture map verified by seniors; root and per-module `CLAUDE.md`; import-linter with current violations ignored; commit `.claude/` with hooks and commands; start the `/plan` workflow |
| **Ongoing**   | Weekly architecture review; burn down ignored imports; module walkthroughs; track the metrics above                                                                                         |
Start strict on limits (300 lines locally, 400 in CI, 80% changed-line coverage) and
loosen only with evidence