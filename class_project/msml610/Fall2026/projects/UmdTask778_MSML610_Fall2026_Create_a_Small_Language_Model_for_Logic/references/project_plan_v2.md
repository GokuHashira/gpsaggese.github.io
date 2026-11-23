# Small Language Models for Math and Logic: Project Plan v2

- **Course:** MSML610 Advanced Machine Learning, Fall 2026 (Small Research Project)
- **Team:** Gokulakannan Sakthivel, Basil Varghese
- **Branch flow:** `gk-dev` → PR into `UmdTask778_MSML610_Fall2026_Create_a_Small_Language_Model_for_Logic_1` (fork only)
- **Plan v2 adopted:** 2026-11-18 (replaces `idea.v1.txt`)
- **Status tracker:** see section 9 (update it as work completes)

---

## 1. Goal

Train small decoder-only language models **from scratch** to be as strong as possible at
grade-school math and multi-step logic, using **real open datasets plus our own
program-verified synthetic data**, and measure how ability depends on model size and on each
training stage (pretraining → SFT → post-training with checkable rewards).

### What changed from v1
| | v1 (done) | v2 (this plan) |
|---|---|---|
| Data | Synthetic only (M1–M3, L1–L3) | Real open math data **+** synthetic |
| Architecture | Llama-style, multi-head attention | **Qwen2.5-style** (grouped-query attention) |
| Sizes | 25M / 125M / 1B | **~125M / 358M (Qwen2.5-0.5B shape) / 1.31B (Qwen2.5-1.5B shape)** |
| Tokenizer | 2k BPE | **32k BPE**, still one token per digit |
| Context | 1,024 | **2,048** |
| Tokens per model | 2.5B | **5B / 8B / 12B** |

The v1 models (25M and 125M, synthetic-only) are kept as **baselines**. The v1 1B run was
cancelled at 53% to save SU for v2.

## 2. Research Questions

1. **Size:** how do accuracy and multi-step generalization scale from ~1M to 1.31B (six sizes),
   and what is the **minimum size for reliable 2–3-step reasoning** (the assigned idea's Q1)?
2. **Stages:** how much does each stage (pretraining, SFT, post-training) add at each size?
3. **Real vs. synthetic data:** do real math datasets improve generalization over synthetic-only
   training (v1 baselines vs. v2)?
4. **Checkable signal:** is post-training with a correctness signal (rejection sampling, verifier
   best-of-N, GRPO) better than SFT alone at matched compute?
5. **Generalization:** longer-than-training problems, held-out templates, real benchmarks
   (GSM8K, SVAMP), and symbolic→English logic transfer.
6. **Logic → arithmetic transfer** (assigned idea's Q2): does training on formal logic improve
   arithmetic word problems, or are they independent skills?
7. **Curriculum:** does an easy → hard ordering beat a fixed mixture at matched size?

The assigned idea (`research/ideas/draft.Create_a_Small_Large_Language_Model_for_Logic.md`)
asks three questions; they map to Q1 (minimum size), Q6 (logic transfer) and Q4 (checkable
signal vs. supervised training at the same compute). Its predicted failure mode, "small models
get individual steps right but lose track of the overall problem state", is what we observed
(section 5.1) and what fix C targets.

## 3. Data

### 3.1 Training sources (all open; licenses checked on the dataset cards)
| Source | License | Size used | Role |
|---|---|---|---|
| FineMath-4+ (HuggingFaceTB) | ODC-By | 32 of 64 shards (~9 GB, ~5B tokens) | Pretraining: real math explanations in English |
| TinyGSM | MIT | 11.8M problems (3.75 GB) | Pretraining + SFT: Python solutions converted to checkable steps (parsed, never executed) |
| OpenMathInstruct-2 (NVIDIA) | CC-BY-4.0 | 2M subset (1.3 GB) | Pretraining mix + SFT |
| MetaMathQA | MIT | 395k | SFT (+ pretraining mix) |
| Orca-Math-200k (Microsoft) | MIT | 200k | SFT (+ pretraining mix) |
| GSM8K train | MIT | 7.5k | SFT |
| Our generators M1–M3, L1–L3 | ours | unlimited, streamed | Pretraining + SFT; logic and checkable steps |

### 3.2 Evaluation sets (never trained on)
| Set | Size | Measures |
|---|---|---|
| **GSM8K test** | 1,319 | Main real-world benchmark |
| SVAMP test | 300 | Robustness of simple word problems |
| MATH-500 | 500 | Stretch goal (harder math) |
| ProofWriter test | — | Real-world English logic (L3 transfer) |
| Our 13 synthetic test sets | 2,000 each | In-distribution, 5–8-step extrapolation, held-out templates |
| FOLIO | ~200 | Optional (gated; needs accepting terms on Hugging Face) |

### 3.3 Hygiene
- **Decontamination:** every training example with a word-level 13-gram overlap with any eval set
  is dropped.
- Synthetic training data excludes all eval/validation prompts (exact match).
- Format: `Problem: … Solution: <steps> #### <answer>` for every source (format v2: real-data
  steps are free text; answers compared by exact numeric value).

### 3.4 Pretraining mixture (token shares, corpus v2)
| FineMath-4+ | TinyGSM | OpenMathInstruct-2 | MetaMathQA + Orca + GSM8K | Synthetic math M1–M3 | Synthetic logic L1–L3 |
|---|---|---|---|---|---|
| 46% | 20% | 19% | 5% (3% + 1.96% + 0.04%) | 5% | 5% |

Measured corpus v2 (2026-11-19): 6.92B training tokens after decontamination — FineMath 5.24B,
OpenMathInstruct-2 0.90B, TinyGSM 0.63B (6.5M problems), MetaMathQA 0.09B, Orca-Math 0.06B,
GSM8K train 1.1M; 32k vocabulary. Shares were rebalanced to these counts so the large model reads
every source at most ~4 times.

Real data 90%, synthetic 10% (decision 2026-11-18: keep synthetic low). Shares are capped so no
source exceeds ~4 passes for any model (data-constrained scaling, Muennighoff et al. 2023); exact
values are re-checked once real token counts are known after tokenization.

Real sources are pre-tokenized once (uint16 token files + document offsets); synthetic data is
generated on the fly. Training blocks are deterministic in `(seed, block index)`, so every size
sees the same data in the same order.

## 4. Models (Qwen2.5-style: RMSNorm, RoPE, SwiGLU, GQA, QKV bias, tied embeddings)
| | Small | Medium | Large |
|---|---|---|---|
| Layers × width | 18 × 768 | 24 × 896 | 28 × 1536 |
| Query / KV heads | 12 / 4 | 14 / 2 | 12 / 2 |
| FFN size | 2,176 | 4,864 | 8,960 |
| Non-embedding params | ~119M | **~358M (= Qwen2.5-0.5B)** | **~1.31B (= Qwen2.5-1.5B / Qwen2.5-Math-1.5B)** |
| Vocab / context | 32k / 2,048 | 32k / 2,048 | 32k / 2,048 |
| Pretraining tokens | 5B | 8B | 12B |

Same layer shapes as the Qwen2.5 models, but with our own tokenizer and trained from scratch
(not compatible with Qwen weights).

## 5. Training Stages
1. **Pretraining** on corpus v2 (section 3.4).
2. **SFT:** loss on solutions only; real instruction data + TinyGSM + synthetic; pass@k each
   epoch; checkpoint selected by pass@k (Cobbe et al.).
3. **SFT + fixes (ablation, decided 2026-11-20):** same recipe from the pretrained models,
   plus targeted fixes for the failure modes found in the pre-SFT evaluation (section 5.1).
   Finalized only after analyzing the plain-SFT evaluations of the small and medium models.
4. **Post-training:** rejection-sampling fine-tuning, verifier + best-of-N (including
   small-generator / large-verifier pairings) and GRPO, **all on all three sizes**.

### 5.1 Pre-SFT failure analysis and fixes (2026-11-20)
| Failure (1.31B model, pre-SFT) | Cause | Fix |
|---|---|---|
| ProofWriter below the "always unknown" baseline (35% vs 45.5%); 73% of answers "unknown" | Logic training data is symbolic / template English; ProofWriter phrasing never seen | **B:** 60k ProofWriter train problems (depth 0–3) with worked proofs built from `allProofs` (`logic_slm_proofwriter.py`) |
| ProofWriter eval used the first 2,000 test rows = depth-0 only | Eval set construction | New eval `real_pw-depth5`: depth-5 test config, 350 per proof depth 0–5 (depths 4–5 deeper than any SFT proof) |
| Extrapolation (5–8 steps) 21–36%, does not improve with size | Operator bookkeeping errors (drops a term, flips a sign); steps only write partial results | **C:** M1 state-tracking steps (`state_steps`): each step rewrites the remaining expression, conventional order; train 1–4 steps, test 5–8 |
| MATH-500: ~1/3 of misses have no final answer | 144–178 completions per model hit the 512-token limit | **E:** MATH-500 evaluated with 1,024 new tokens (`eval_model.sbatch` part `math`) |
| Synthetic English logic with held-out templates: 45% | Template brittleness | Covered partly by B; revisit after plain SFT |
| Optional | Length generalization | **D:** self-improvement curriculum (majority-vote-filtered self-generated longer problems) |

Also: bootstrap confidence intervals and 2–3 sampling seeds for the final models.

## 6. Evaluation
Exact-match accuracy, pass@k, accuracy by number of steps, step validity (synthetic sets),
extrapolation, held-out templates, GSM8K / SVAMP / MATH-500, ProofWriter transfer.
Every model is evaluated **after each stage** with the same `evaluate.py`.

## 7. Experiments
| # | Experiment | Answers |
|---|---|---|
| E1 | v1 synthetic-only baselines (25M, 125M) | done: baseline for Q3 |
| E2 | v2: 3 sizes × 3 stages | Q1, Q2 |
| E3 | Extrapolation, held-out templates, real benchmarks, logic transfer | Q5 |
| E4 | v1 (synthetic-only) vs. v2 (real + synthetic) | Q3 |
| E5 | RFT vs. verifier best-of-N vs. GRPO at matched compute | Q4 |
| E6 | Generator / verifier size pairings | TinyGSM finding |
| E7 | Optional: architecture comparison at ~10M (Qwen2 vs. Llama vs. Phi block) | design choice |
| E8 | **Size sweep down to ~1M** (assigned idea): ~1M / 10M / 50M, same recipe, pretrain + SFT, accuracy by number of steps | Q1 (minimum size) |
| E9 | **Curriculum ablation** (easy → hard vs. fixed mixture) at ~10M and ~50M | Q7 |
| E10 | **Logic-transfer ablation** (with vs. without logic data) at ~10M and ~50M | Q6 |
| E11 | **Matched-compute control**: plain SFT continued for the GPU time GRPO uses (358M) | Q4 |
| — | Comparison with the sibling ideas (Kids_Book, Python): literature only (e.g. TinyStories) | assigned idea's research topic |

### 7.1 Small-model sweep (added 2026-11-21, assigned idea)
- **Architecture: unchanged** (Qwen2-style, same 32k tokenizer, same data), so the size curve
  isolates size. Head width 64 (32 at ~1M), grouped-query attention, depth/width in the usual
  range, higher peak LR for smaller models (~1e-3 – 2e-3).
- **Sizes are non-embedding parameters** (as for the other sizes). With a 32k vocabulary the
  tied embedding adds ~4M / ~10M / ~18M, so totals are reported too.
- **Tokens:** ~0.5B (1M), ~1B (10M), ~2B (50M); cheap, and past the compute-optimal point.
- **Curriculum (E9):** synthetic difficulty (number of steps) and the real-data mix move from
  easy (GSM8K-style) to hard (MATH-style) over training; same tokens as the fixed mixture.
- Curriculum and no-logic stay at ~10M / ~50M only (team decision 2026-11-22: not repeated for the main models).
- **No-logic (E10):** the logic generators removed from the mixture (shares renormalized);
  arithmetic test sets compared with the matching E8 model.
- GPUs: A100 when available (48 SU/h), else H100.

## 8. Resources

### 8.1 Compute (Zaratan, account `msml610-class` only; H100 = 144 SU/GPU-hour)
| Item | Est. SU |
|---|---|
| Already spent on v1 (benchmarks, 25M, 125M, cancelled 1B, tests) | ~2.9k |
| Benchmark of v2 configs | ~0.1k |
| Pretraining: small 5B tokens / medium 8B / large 12B | ~0.4k / ~1.8k / ~9.7k |
| SFT, all sizes | ~1.0k |
| Post-training (RFT + verifier all sizes, GRPO medium) | ~2.5k |
| Ablations + evaluation | ~1.0k |
| **v2 total (excluding v1)** | **~16.5k** |

**Re-estimate (2026-11-20), every stage on all three sizes.** Measured so far: pretraining
~13.3k SU; plain SFT runs at ~1.1 / 1.6 / 3.2 s per step (≈ 3.5 / 5 / 11 GPU-hours).
Our user had used ~15.8k SU of the class pool by 2026-11-20.

| Remaining (all 3 sizes) | Est. SU |
|---|---|
| Plain SFT (rest of the running jobs) | ~2.6k |
| SFT + fixes (3 epochs; ~1k less with 2) | ~3.0k |
| RFT | ~1.0k |
| Verifier + best-of-N | ~0.85k |
| GRPO | ~1.75k (least certain; re-estimated after the small run) |
| Evaluations after every stage | ~0.9k |
| **Remaining total** | **~10–11k** → project total ≈ 26–27k SU |

**Assigned-idea experiments (added 2026-11-21):**

| Item | Est. SU |
|---|---|
| E8 size sweep: pretraining ~1M / 10M / 50M | ~150–250 |
| E8 SFT of the three small models | ~100 |
| E8 evaluations (3 sizes × 2 stages) | ~150 |
| E9 curriculum (~10M, ~50M) incl. evaluation | ~250 |
| E10 no-logic (~10M, ~50M) incl. evaluation | ~250 |
| E11 matched-compute control (358M) | ~400–500 |
| **Added total** | **~1.3–1.5k** → project total ≈ 27–28k SU |

Class pool on 2026-11-18: 50 kSU limit, **45.2 kSU unused** (shared by the whole class).
v2 uses ~35% of what is left: feasible, but give the TA a heads-up before the large run.

### 8.2 Storage (class quota 150 GB, ~60 GB free, shared)
| Item | Size |
|---|---|
| Raw downloads (deleted after tokenization) | ~18 GB (peak) |
| Tokenized real data (uint16) | ~15 GB |
| Checkpoints (all sizes and stages, bf16, pruned) | ~10–12 GB |
| **Steady state** | **~27 GB** |

All project files live under `/scratch/zt1/project/msml610/user/gsakthiv` only.

**Data retention (decision 2026-11-19):** keep the raw downloads (`data/raw/`, ~14 GB), the
converted data (`data/processed/real/`) and corpus v2 for the whole project, so any converter,
tokenizer or corpus fix can be rebuilt. Nothing is deleted unless storage actually runs short,
and only with the team's explicit approval.

### 8.3 Time (deadline ~mid-December)
| Phase | Target |
|---|---|
| Data download, conversion, decontamination, tokenizer, corpus v2 | Nov 19–21 |
| Benchmark + pretraining (small, medium, large; large ≈ 17 h on 4 H100s) | Nov 21–24 |
| SFT (all sizes) | Nov 24–27 |
| Post-training + ablations | Nov 27 – Dec 5 |
| Notebooks, README, Docker, report, video | Dec 5–13 |

## 9. Status Tracker
- [x] Format v1, synthetic generators M1–M3 and L1–L3 (z3-checked), scorers, tokenizer v1
- [x] Streamed pretraining, DDP, W&B/TensorBoard, stop/resume, eval-only, SFT script
- [x] v1 pretraining: 25M and 125M done (98–100% in-distribution); 1B cancelled at 53%
- [x] Downloads (login node; compute nodes cannot reach the Hugging Face file servers)
- [x] Converters: TinyGSM → steps (6.5M kept of 11.8M); OpenMathInstruct-2 1.98M, MetaMathQA 393k, Orca-Math 165k, GSM8K 7.5k
- [x] 13-gram decontamination against all eval sets (incl. FineMath: 2,056 docs removed)
- [x] Qwen2.5-style model configs (~119M / 358M / 1.31B)
- [x] 32k tokenizer (single digits, reserved symbols)
- [x] Corpus v2 (6.92B tokens, tokenized real sources + streamed synthetic)
- [x] `evaluate.py` (greedy + pass@k + majority vote + cost) and run ledger `results/runs.jsonl`
- [x] H100 benchmark: 295k / 144k / 54k tok/s per GPU → pretraining ≈ 11.7k SU
- [x] Pretraining: small / medium / large (val loss 1.349 / 1.236 / 1.140; ~13.3k SU)
- [x] Pre-SFT evaluation of all three sizes; failure analysis (section 5.1)
- [x] Plain SFT: small / medium / large (3 epochs; best = epoch 3 for all; 358M epoch 3 resumed after a scoring crash)
- [x] Fix code: ProofWriter proofs + depth-5 eval, M1 state-tracking steps, MATH-500 at 1,024 tokens (CPU-tested)
- [x] Re-evaluate pretrained models on MATH-500 (1,024 tokens) and ProofWriter depth-5
- [x] Evaluate plain-SFT models; analyze; share the analysis with the team; finalize fixes (approved 2026-11-21: B, C, 120k synthetic, per-source validation, 3 epochs)
- [ ] SFT + fixes: 119M ✅ trained + evaluated (ProofWriter d5 8.1 → 60.4, arithmetic 5–8 steps 16.6 → 58.8); 358M ✅ and 1.31B ✅ trained (evaluating); resumed after the 2026-11-21 quota crash
- [ ] E8 size sweep (~1M / 10M / 50M): configs ✅, pretraining ✅ (val loss 2.434 / 1.815 / 1.514), evaluation ✅ (GSM8K 2.1 / 13.4 / 26.8%), SFT + fixes ✅ trained (evaluating)
- [ ] E9 curriculum ablation, E10 no-logic ablation (~10M, ~50M): pretraining ✅, evaluation ✅ (curriculum: no gain; no-logic: math slightly better, no transfer) (small sizes only, by team decision)
- [ ] E11 matched-compute control (with GRPO)
- [ ] Optional E7 architecture comparison at ~10M
- [ ] Storage: class quota raised to 250 GB on 2026-11-23 (was 150 GB, shared by the class); raw downloads moved to the team's laptop (restore with rsync); keep only `best/` checkpoints for finished stages; quota guard in job launchers
- [ ] Post-training: RFT, verifier best-of-N, GRPO
- [ ] Ablations, notebooks, README, Docker, report, video
- [ ] Publish on Hugging Face (after GRPO): model repos + model cards (eval tables, cost, limits), free CPU Space demo; first check course permission and TinyGSM's license
- Running story of the project (what failed and how it was fixed): `references/project_log.md`

## 10. Risks
| Risk | Mitigation |
|---|---|
| Shared class SU pool drained by others | Run the large pretraining first; check `sbalance` after each stage |
| Shared class storage fills up | Delete raw files after tokenization; bf16, pruned checkpoints |
| GPU queue delays | Submit early; H100 or A100 by availability |
| Small models stay weak on GSM8K | Verifier best-of-N and RFT (TinyGSM's key finding); report honestly |
| Data contamination | 13-gram decontamination + exact-prompt exclusion |
| Job interruptions | Checkpoints, clean stop before the time limit, self-resubmission |
