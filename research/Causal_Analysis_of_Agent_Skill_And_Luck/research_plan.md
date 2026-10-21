# Causal Analysis of Agent Skill and Luck

## Milestone 1 Research Plan

Current plan: the dated update below contains the confirmed Deal or No Deal direction
and current implementation status. I have retained the earlier proposal for context;
the update supersedes its pending scope decisions and older progress statements

# Goal

My goal is to bring our work into one paper addressing a shared question: how can we
distinguish underlying ability and decision quality from favorable circumstances and
random outcomes, and use that distinction to evaluate success more reliably?
I propose two separate studies under this question. Niveda will investigate skill and
luck in coding-agent benchmark evaluation. Riya and Stuti's original proposal extends
my societal success simulation to study learning, networks, and resource allocation
Each study will retain its own model and experiments, while contributing to a common
framework and discussion in the paper
There is one scope decision to resolve before Riya and Stuti begin substantial new
implementation. GP's
[latest comment on \#659](https://github.com/gpsaggese/gpsaggese.github.io/issues/659#issuecomment-6290190159)
recommends considering
[Deal or No Deal (\#672)](https://github.com/gpsaggese/gpsaggese.github.io/issues/672)
instead of the original extensions, which he describes as incremental. I include
their original plan below and a concrete replacement plan if we take his
recommendation. I will confirm that choice with the GP and the team; we should
execute one of those options rather than commit to both

# What Is Done

My
[previous project](https://github.com/bkk8403/A-Causal-Analysis-of-Success-in-Modern-Society)
provides the societal baseline: an agent-based simulation with intensity, IQ,
networking, capital, and random events, together with inequality metrics, causal
estimation, and allocation-policy comparisons. The
[current paper](https://github.com/gpsaggese/gpsaggese.github.io/tree/master/papers/Causal_Analysis_of_Agent_Skill_And_Luck)
describes this model and a broader mathematical framework. I want us to reproduce its
reported results before extending or relying on them
Niveda has implemented a separate synthetic benchmark simulator in skill_luck_sim/
It models configuration ability, task difficulty, configuration–task interactions,
and repeated binary outcomes. It includes tests, saved results, and plots for
winner's curse, ranking errors, and fresh-seed versus fresh-task reevaluation
[PR \#605](https://github.com/gpsaggese/gpsaggese.github.io/pull/605) was merged into
the review branch. Its parameters are assumed rather than fitted to real agent runs;
the real benchmark pilot and integration into the shared research structure remain
next steps
Riya and Stuti have
[proposed](https://github.com/gpsaggese/gpsaggese.github.io/issues/659) changing
skills over time, explicit connection networks, and adaptive allocation, followed by
possible real-data validation. I treat these as proposed extensions rather than
completed implementations. Their individual division of work still needs agreement
Our immediate technical gaps are reproducibility, consistent definitions, and
integration. Niveda has reported differences between her societal runs and the paper,
skipped tests and old imports, and a performance-based allocation policy that can
match equal allocation when starting capital is identical. The paper's numerical
findings should remain provisional until we explain those differences

# Next Steps

## Shared Foundation

By shared foundation, I mean the definitions, working code, and experiment records
that both studies need before we build on their results. Each simulator will keep its
own model, baseline, and experiments. The foundation connects how we describe and
validate the studies; it does not combine them into one simulator
First, we need to agree on the meaning of skill, expected performance, and luck in
each study. We will use consistent terminology and symbols while keeping the outcomes
distinct: capital or another specified success measure in the societal model, and
task resolution or mean score in the benchmark model. For each expected outcome, we
will state what is held fixed and what is randomized. We will also distinguish
residual luck—observed outcome minus expected outcome—from the paper's separate
event-level score based on utility, surprise, and control
Second, we will establish a reliable starting implementation for each study. We will
agree on the reference code versions and integration branch, restore test collection
and imports, and reproduce the existing reference results. Niveda will lead the
cleanup assigned by GP, and I will contribute to baseline reproduction and
validation. We will document configurations, seeds, software versions, and commands
so another contributor can rerun the experiments
We need this foundation for two reasons. Within each study, it lets us distinguish an
effect of a new mechanism or evaluation method from a bug, a changed setting, or an
unreproduced baseline. Across the studies, it lets us explain precisely which ideas
are shared and why both belong in the same paper, without implying that their
outcomes or numerical results are interchangeable. Any unexplained discrepancy will
be recorded and its impact assessed before we use the affected result to support a
conclusion
This step is complete when we have a short definition of the quantities and reference
conditions for each study, runnable tests and reference experiments, and saved
results with reproduction instructions. Discrepancies must either be resolved or
explicitly tracked with a clear statement of which conclusions remain unsupported

## Plan for Niveda

Research question: how much of a coding-agent leaderboard score reflects persistent
performance, how much depends on sampled tasks and execution randomness, and how much
optimism is introduced when we select the best observed configuration?
**N1 Validate the synthetic baseline:** Reproduce the saved reference experiments and
test the distinction between new execution seeds on fixed tasks and genuinely new
tasks. Define expected performance for each target. Evaluate score inflation for the
selected winner, probability of selecting the truly best configuration, and regret
Validate selected-winner uncertainty explicitly, with shared tasks preserved in
resampling. Output: reproducible baseline figures and a short explanation of the
assumptions and limits
**N2 Run the feasibility pilot:** After GP confirms model access and a capped budget,
choose one runnable open-source agent and one fixed configuration. Run 20–30
SWE-bench Verified tasks with five independent repeats per task, giving 100–150
attempted task runs before retries. Save task IDs, binary outcomes, versions,
budgets, execution settings, costs, and failure categories. Specify how
infrastructure failures and retries are handled. Output: a replayable outcome dataset
and a variability/cost report. This pilot measures repeatability; it cannot establish
selection bias across configurations
**N3 Compare configurations with paired evaluation:** Use the pilot to choose an
affordable number of agents or scaffold configurations and tasks, rather than
immediately committing to the idea document's full 2–3 benchmark, 3–5 agent matrix
Evaluate configurations on the same tasks, separate task-sampling from run
variability, and report paired score differences with confidence intervals. Estimate
the number of tasks needed to detect a prespecified meaningful gap. Output: an
uncertainty-aware comparison stating which differences the evidence can resolve
**N4 Test selection and correction:** Prespecify a candidate search, its budget, and
a task split. Select a configuration using development tasks, freeze it, and evaluate
on independent held-out tasks. Compare naive winner reporting, same-task fresh-seed
reevaluation, and fresh-task evaluation. Vary search size first in simulation and
then within the available real-run budget. Treat square-root-log search penalties as
approximations to test, not universally valid corrections. Output: measured optimism
and a validated reporting protocol with its cost and limitations
**N5 Extend only after the core result:** If the data and budget support it, test
rank persistence across task splits and a second benchmark, then consider item
response theory to estimate task difficulty and discrimination. A meaningful IRT
analysis requires substantially more configurations than the one-agent pilot
Adaptive task selection and contamination analysis remain later extensions. Output:
evidence about transfer or informative tasks, if feasible, without delaying the core
paper
I propose that Niveda develop the state-of-the-art document alongside N1 and N2
[Fama–French](https://mba.tuck.dartmouth.edu/bespeneckbo/default/AFA611-Eckbo%20web%20site/AFA611-S8C-FamaFrench-LuckvSkill-JF10.pdf)
motivates comparing observed winners with chance-based reference distributions;
[Bailey and colleagues](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
motivate evaluating the selection process;
[SIREN](https://arxiv.org/html/2605.05973v1) is a direct comparator for
selection-aware LLM reporting; [SWE-rebench](https://arxiv.org/html/2505.20411v1)
motivates repeated, standardized evaluations. We need to identify what our
coding-agent experiments add beyond these existing results

## Plan for Riya and Stuti

Original research question: when ability, social connections, and allocation rules
can change over time, how does that change the relationship between underlying
ability, opportunity, and accumulated success?
**R1 Reproduce and clarify the societal baseline:** Run the agreed configuration over
independent seeds and regenerate inequality, wealth-share, and skill–outcome results
Separate exposure to an opportunity from successfully exploiting it; successful-event
counts already depend on ability. Clarify allocation timing and outcome scale. Check
causal estimates against explicitly defined simulator interventions rather than
assuming that a fitted regression establishes causality. Output: a baseline report
and a list of resolved or outstanding paper/code differences
**R2 Test learning and skill decay first:** Specify one learning/decay mechanism and
how it affects opportunity exploitation or exposure. Keep it optional so zero
learning and zero decay recover the baseline. Compare the fixed-skill model with the
extension using matched initial agents and seed sets, and report uncertainty across
independent worlds. Measure inequality, mobility, and how observed success ranks
agents relative to their expected performance. Output: one controlled extension with
an interpretable result, including a null result if that is what we find
**R3 Add networks as a separate experiment:** Replace or extend the existing scalar
networking mechanism with an explicit graph. Compare a small set of justified graph
structures while controlling overall opportunity supply and other assumptions. The
existing model already makes exposure and spillovers depend on characteristics; the
new contribution is topology. Output: evidence about whether network structure
changes the persistence and concentration of advantage
**R4 Evaluate adaptive allocation:** After the previous results are stable, compare a
specified adaptive policy with fixed-policy baselines using equal total budgets and
evaluation horizons. Any policy based on past performance must observe an initial
history before allocating. Evaluate welfare, inequality, and the risk of repeatedly
rewarding fortunate agents. Output: a policy comparison with clear timing,
information, and resource constraints. Real-data calibration follows only after a
suitable dataset and validation target are established
I propose that Riya and Stuti choose a primary implementer and a reviewer for each
task and rotate those roles where useful. I will support the baseline and review the
interpretation. Their published issue is a joint proposal, so I will not assume an
individual split before we agree on it
[Pluchino and colleagues](https://arxiv.org/html/1802.07068v3) provide the simulation
precedent; our contribution must be a defined mechanism or experiment beyond
reproducing the existing finding that luck can matter

## If We Follow GP's Deal or No Deal Recommendation

I propose replacing R2–R4 with a focused decision-quality study, retaining my
societal model as existing background or a reproduced case study. Riya and Stuti
would obtain round-level contestant data, reproduce one analysis from the relevant
literature, implement an optimal stopping benchmark with an explicit model for future
banker offers, and then incorporate risk preferences. They would compare observed
choices with that benchmark and investigate whether previous gains or losses predict
departures after accounting for the current game state
The first deliverable would be a cleaned, documented dataset and an end-to-end
example producing a benchmark decision for a small set of rounds. The next
deliverable would be a validated analysis of decision quality and sensitivity to the
risk and offer models. We must distinguish a reasonable decision followed by bad luck
from a poor decision, and distinguish descriptive associations from identified causal
effects
[Post and colleagues](https://www.aeaweb.org/articles?id=10.1257/aer.98.1.38) are the
central empirical reference. This option is pending the scope decision; I am not
assigning both it and the original extension programme

## How the Studies Unify Into One Paper

I propose a common framework based on observed outcome, expected performance under a
specified reference environment, and the residual around that expectation. Repeated
simulations or evaluations let us estimate uncertainty in expected performance; a
finite-sample estimate is not the true expectation. Predictable advantages may
include circumstances and resources, so we should not equate them automatically with
earned skill. Statistical decomposition alone does not establish a causal
attribution
| Element          | Niveda study                                                                   | Riya and Stuti original study                                                   |
| :--------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| Outcome          | Task resolution or mean benchmark score                                        | Final capital, inequality, or mobility                                          |
| Reference        | Expected performance of a fixed configuration on a specified task distribution | Expected outcome under a defined agent profile or policy across repeated worlds |
| Random variation | Sampled tasks, execution randomness, and logged infrastructure effects         | Event histories and graph/event realizations under controlled assumptions       |
| Decision tested  | Select a configuration and report its performance                              | Allocate opportunities or resources using defined information                   |
| Shared evidence  | Bias, rank stability, uncertainty, and selection regret                        | Rank stability, uncertainty, welfare, and inequality                            |
The shared contribution I want to test is whether decisions based on one realized
success measure differ from decisions informed by repeated performance and
uncertainty. In Niveda's study the decision is which configuration to select and
report. In the societal study it is whom to reward or fund. Where feasible, I propose
comparing realized-outcome ranking with an expected-performance reference in both
studies. If we instead choose Deal or No Deal, the link becomes decision quality
versus realized payoff: the benchmark study evaluates systems, while the game-show
study evaluates choices under uncertainty
The paper will use a shared introduction and framework, separate methods and results
for each study, and a joint discussion of evaluation and decision-making under
uncertainty. I propose reporting each study's uncertainty and limitations separately
We will not transfer a percentage of luck from one domain to another, label
observational residuals as proven causal luck, or promise a new correction before it
is validated. Negative findings can still explain when a method or proposed mechanism
does not help

## How We Will Proceed

I have reviewed Niveda's plan and Riya and Stuti's proposal. My next step is to
confirm the research scope with GP and the team, including whether Riya and Stuti
continue the societal simulation extensions or take up Deal or No Deal. We will agree
on the questions each study will answer and how its results will contribute to the
paper
Next, we will agree on the shared definitions and reproduce the existing results for
each simulator. Niveda will lead code cleanup, and I will contribute to baseline
validation. The literature review can proceed alongside this work so that we identify
what each study adds to existing research
Once the relevant setup is verified, the studies can proceed in parallel. Niveda will
run the small coding-agent pilot after the evaluation protocol and budget are
confirmed. Riya and Stuti will carry out the first experiment in their agreed
direction. Neither study needs to wait for unrelated work in the other
We will review the initial results with GP before expanding the experiments. For
Niveda's work, this means deciding how to compare multiple configurations and
evaluate the selected configuration on independent tasks. For Riya and Stuti's work,
it means deciding which further experiment is needed to answer their research
question. We will also check whether both studies provide evidence for the paper's
proposed common argument
Finally, we will build the paper around the validated findings as they become
available. Each study will have its own methods, results, and limitations. I will
coordinate the shared framework and discussion so that the paper explains what the
studies establish together, where they differ, and which questions remain open
I will coordinate the paper outline, research decisions, and integration, and
contribute to baseline validation. Niveda will lead the benchmark study and assigned
cleanup and literature work. Riya and Stuti will lead their agreed study. GP will
supervise the research scope and scientific claims. We will agree on individual task
owners and milestones, then track the immediate work in GitHub issues linked to this
plan

# Update: 20 October 2026

Riya and Stuti have confirmed that they will take up Deal or No Deal. This updates
the earlier plan: their work will focus on decision-making under risk rather than the
proposed learning, network, and adaptive-allocation extensions to my societal
simulation. The earlier sections are retained as the original proposal; this update
supersedes those assignments and the statements that their direction is still
undecided

## Current Implementation Status

Niveda's benchmark simulator has now been integrated through PR
[\#695](https://github.com/gpsaggese/gpsaggese.github.io/pull/695), merged on 20
October into UmdTask537_Review_students_research_projects. It now lives in
research/Causal_Analysis_of_Agent_Skill_And_Luck/ as agent_benchmark_utils.py,
notebooks, and tests. The PR reports 21 passing benchmark tests and unchanged outputs
for the same seeds; it does not establish completion of the real-agent pilot or
integration into the default branch
PR [\#709](https://github.com/gpsaggese/gpsaggese.github.io/pull/709) is open for the
societal utilities and test cleanup. Niveda reports 74 tests passing with PyMC
installed and unchanged notebook results. Her correction in
[\#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707) identifies the
PyMC import skip and outdated imports in the test file. The proposed tests still skip
when PyMC is unavailable, and the PR reports unresolved dependency-installation
conflicts. These are author-reported checks, not results I have independently rerun
We still need a reproducible environment, full test discovery, review of the
notebook/README notation work, and reproduction or correction of the paper's
numerical results before calling the cleanup complete
The real-agent pilot and the state-of-the-art document remain proposed work. I will
track the literature work in
[\#706](https://github.com/gpsaggese/gpsaggese.github.io/issues/706) and
notation/reproducibility work in
[\#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707), alongside the
remaining cleanup tasks under
[\#541](https://github.com/gpsaggese/gpsaggese.github.io/issues/541). Merging a
refactor does not resolve the reported Gini, event-count, or allocation-policy
discrepancies by itself

## Updated Plan for Riya and Stuti

Their research question is how to distinguish the quality of a contestant's decision
from the luck of the resulting payoff. They will compare actual deal/no-deal choices
with a decision benchmark conditional on the information available at that round and
explicitly stated risk-preference assumptions
First, they will obtain and document round-level contestant data and reproduce a
relevant analysis from the existing literature. The initial output will be a cleaned
dataset containing the remaining case values, banker offers, and observed choices,
together with an end-to-end example
The data will identify contestants, country editions, round schedules, actual
decisions, and any hypothetical play after acceptance. Only actual decision rounds
will enter the stopping-choice analysis. They will account for the fact that later
rounds are observed only for contestants who continued. Future-offer models will be
fitted and assessed on separate games or contestants; the evaluated decision will use
only information available at that round, without later realized offers or case
openings
Next, they will implement and validate an optimal stopping benchmark under explicit
assumptions about future banker offers. They will begin with risk neutrality, then
examine how the recommended choices change under different risk preferences. The
output will be benchmark decisions and expected values or utilities for the observed
game states
Risk preferences will first be treated as prespecified sensitivity scenarios. If Riya
and Stuti estimate preference parameters from choices, they will state the
identification assumptions, report uncertainty or compatible parameter ranges, and
assess predictions on decisions excluded from estimation. A single choice may not
identify a contestant's preferences. Model-based utility regret will compare the
observed action with the best action under the same stated preferences and offer
model; it will not be presented as an observed counterfactual loss
They will then compare observed choices with the benchmark, investigate whether
earlier gains and losses are associated with later decisions after accounting for the
current state, and test sensitivity to the offer and risk-preference assumptions. A
low realized payoff will not by itself count as evidence of a poor decision. Any
causal claim will require a separate identification argument

## How the Updated Studies Connect

The common paper question becomes: how can we evaluate performance and decision
quality without confusing favorable outcomes with skill or sound choices? Niveda's
study examines whether a coding-agent configuration looks better because of favorable
task samples or execution randomness. Riya and Stuti's study examines whether
contestants made reasonable choices given the risks, even when the eventual payoff
was unfavorable
I propose a shared framework with three quantities: the realized outcome, the
expected outcome conditional on the relevant configuration or choice and information,
and the difference between those two. Decision quality is a separate comparison
between the expected value or utility of the chosen action and the best available
alternative under that study's assumptions. Finite-data estimates carry uncertainty;
expected performance can include favorable resources and circumstances as well as
ability
Both studies will ask when conclusions drawn from realized success change after
accounting for the relevant conditional reference and uncertainty. Niveda will
measure selected-configuration optimism, ranking stability, and selection regret
using repeat runs and independent tasks. Riya and Stuti will measure model-based
decision disagreement and conditional utility regret, and explain why realized
payoffs can tell a different story. Their assessment depends on the preference and
offer models, whereas Niveda can empirically reevaluate a frozen configuration. We
will keep those evidential differences explicit and will not pool the outcomes into a
universal percentage of luck or infer causality from a residual
The paper will contain a shared problem statement and framework, a benchmark
methods/results section, a Deal or No Deal methods/results section, and a joint
discussion of the evidence, assumptions, and limits. Each team will reproduce a
relevant published result and identify a specific contribution beyond it before
expanding the study. The common contribution is a tested account of how outcome-based
evaluation differs from evaluation using conditional expectations and uncertainty,
rather than a claim that one estimator applies everywhere
My societal success work remains prior methodological background and, where
reproduced and useful, an illustrative case study. Results from it must be reproduced
or corrected before supporting the paper's claims. It is no longer Riya and Stuti's
extension roadmap. GP's Law remains another possible research direction, outside the
two current studies

## Immediate Sequence and Coordination

First, I will review the two confirmed directions and proposed common argument with
GP. Niveda will verify the integrated benchmark baseline and pursue the remaining
cleanup and notation work, while Riya and Stuti identify a usable Deal or No Deal
dataset and reproduce a published example. I will contribute to the societal baseline
reproduction and paper/code discrepancies. The two studies can proceed independently
where their dependencies allow
Next, Niveda will run the one-configuration pilot after the execution protocol and
capped run budget are agreed. Riya and Stuti will validate their stopping benchmark
and preference assumptions before analyzing deviations. We will review these initial
outputs with GP, then choose the configuration-comparison and selection experiments
for Niveda and the validated decision-quality analysis for Riya and Stuti. We will
write the methods and results as the evidence becomes available
I will coordinate task ownership, answer questions, review dependencies, and maintain
the common paper outline. Niveda will lead the benchmark work; Riya and Stuti will
agree on an implementer and reviewer for their next task, rather than assuming an
individual split here. GP will supervise the research decisions and scientific
claims
I will use [\#705](https://github.com/gpsaggese/gpsaggese.github.io/issues/705) for
this plan, [\#706](https://github.com/gpsaggese/gpsaggese.github.io/issues/706) for
the state of the art,
[\#707](https://github.com/gpsaggese/gpsaggese.github.io/issues/707) for notation and
reproducibility, and
[\#672](https://github.com/gpsaggese/gpsaggese.github.io/issues/672) for Deal or No
Deal. I will agree the remaining task owners and dates with the team, collect
availability for a recurring stand-up, and use the research project to track agreed
tasks without duplicating existing issues. The shared project-folder placement,
stand-up time, and detailed issue breakdown remain coordination steps; preparing this
document alone does not complete all of Milestone 1\. After the paper is reviewed and
its findings are validated, we can proceed to GP's publication and blog goals in
[\#541](https://github.com/gpsaggese/gpsaggese.github.io/issues/541)