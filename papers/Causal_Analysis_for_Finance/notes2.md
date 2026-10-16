# Goal
  - Build a tool to be used (partially) automate idea generation and refinement
  - The goal is not to remove humans from the loop, but automate the boring part
    - Humans to supervise and direct
  - Difficult to automate the entire flow
    - The goal is automate 80-90% -> speed up = ...

# Workflow

## Build Paper Bank
- On-board papers / books / ...
  - Human guides the search
- Parse papers from PDF into markdown

- Filter based on:
  - Keywords (e.g., microeconomic, macroeconomic)
  - Regions (e.g., APAC, EMEA, US)
  - Citations
  - Quality of the researchers

- Extract causal claims
  - (Cause, effect, sign)
  - Context
  - Annotate with reference for human analysis

- Normalize variables
  - E.g., monetary tightening, policy rate hike, increase in federal fund rate
  - Use specialized LLM (trained on finance literature)
  - Review by human
  - Create operators (increase, delay, ...)

- How to handle equilibrium?
- How to handle delays

- Create DAG with
  - Nodes: variables
  - Edges: relationships
  - Enforce DAG-ness (e.g., remove feedback loops)

- Map nodes to timeseries to data store (e.g., Atlas, ...)
  - Assess quality of each timeseries
    - E.g., coverage, time span, gaps

- DAG "reduction"
  - Not all the edges are meaningful
    - Test for causality / association / ...
  - Not all the nodes are predictive or have high-quality data
  - Propose missing nodes and edges from LLM or human knowledge

- Optimize against a proxy
  - E.g., use model to predict supply and demand, instead of prices
  - Regularize to control complexity vs predictive power

- Formalize set of hypothesis to test
  - Create a Bayesian prior from the DAG
  - Fit
  - Estimate uncertainty

## Validation
// This is the official flow ADIA uses to validate a proposal
- Create a proposal

- Vote on the proposal

- Run backtesting
  - Use 
  - Deflate / adjust by tested hypothesis

- Final gate
  - Run OOS
