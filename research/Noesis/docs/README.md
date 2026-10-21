# Docs

- All docs of `research/Noesis`, grouped by kind
- Start with `onboarding.md`, then `architecture.md`
- Component READMEs live next to the code: `../checker/`, `../common/`, `../gateway/`,
  `../market/`, `../noesis_platform/`

## Structure of the Dir

| Dir         | Description                                                    |
| :---------- | :------------------------------------------------------------- |
| `findings/` | Raw measurements, e.g., the OpenRouter provider spike          |

- All other docs sit at the top level of `docs/`, grouped by the cluster in the
  table below

## Description of Files

| File                           | Description                                                                           | Cluster    |
| :----------------------------- | :------------------------------------------------------------------------------------ | :--------- |
| `architecture.md`              | C4-model architecture (context, container, component) of the code                     | Design     |
| `checker_design.md`            | Checker evaluation flow, window policy, and what is not built                         | Design     |
| `gateway_design.md`            | Gateway design: request flow, blame rules, faults, endpoints, config                  | Design     |
| `glossary.md`                  | Terms used across the code and docs                                                   | Onboarding |
| `how_we_work.md`               | Branch, PR, test, and review conventions                                              | Onboarding |
| `onboarding.md`                | Setup, run, test, and first tasks                                                     | Onboarding |
| `plan.noesis.md`               | Roadmap (`PR_M*`, `PR_S*`, `PR_P*`) and MVP execution plan (milestones, owners, tasks) | Plans      |
| `progress.noesis.md`           | Live MVP status and history                                                           | Plans      |
| `plan.marketing.md`            | Bootstrapping supply and demand liquidity                                             | Plans      |
| `mvp_prd.md`                   | MVP product requirements                                                              | Specs      |
| `decisions.noesis.md`          | MVP decisions and the alternatives considered                                         | Specs      |
| `spec.PR_M16.md`               | Implementation spec: economic surplus of Noesis vs a posted-price baseline            | Specs      |
| `spec.PR_P2.md`                | Implementation spec: containerized cloud deployment                                   | Specs      |
| `spec.PR_P2b.md`               | Implementation spec: Postgres backend                                                 | Specs      |
| `dev_notes.md`                 | Build and OpenAPI-generation notes                                                    | Notes      |
| `gp_todos.md`                  | Open TODOs                                                                            | Notes      |
