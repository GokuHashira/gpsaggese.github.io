# Running Our Small Math Models on a Phone

As of 2026-11-26. Owner: Basil. Not started yet. This is the project's live demo (no hosted GPU demo is planned).

## 1. Goal

Run our trained math and logic models fully offline on a phone and show it live in the final
presentation. The 358M model compressed to 4-bit is about 250 MB and should answer a GSM8K-style
problem in a few seconds with no network.

This fits the project's core question, how good small models can get: a model that runs in airplane
mode on a phone is the practical payoff of staying small.

## 2. Models and phone footprint

All six models fit on a recent phone once compressed to 4-bit. The 358M is the sweet spot for the main
demo; the full set lets the audience see the size sweep on the phone itself. Sizes are estimated from
total parameter counts (the shared 32k-token embedding dominates the smallest models); speeds are
approximate for current phones and must be measured.

| Model | Parameters (total) | 16-bit file | 4-bit file | Expected speed | 100-token answer | GSM8K (SFT + fixes) |
|---|---|---|---|---|---|---|
| 1M | ~5.2M | ~10 MB | ~4 MB | very fast | well under 1 s | 1.9% |
| 10M | ~20.5M | ~41 MB | ~13 MB | very fast | under 1 s | 11.0% |
| 50M | ~69.2M | ~140 MB | ~45 MB | 150+ tokens/s | under 1 s | 24.9% |
| 119M | ~144M | ~290 MB | ~90 MB | 100+ tokens/s | about 1 s | 36.2% |
| 358M | ~387M | ~775 MB | ~250 MB | ~50 tokens/s | about 2 s | 41.3% |
| 1.31B | ~1.36B | ~2.7 GB | ~0.8 GB | 15–30 tokens/s | 3–7 s | 46.4% |

The accuracy column is from the cluster evaluation of the SFT + fixes models (greedy, 16-bit). The
checkpoint to ship is the GRPO model of each size (GSM8K 2.9 / 17.3 / 27.4 / 38.4 / 42.3 / 46.3% for
1M ... 1.31B; for the 1M model the folder is `models/grpo2_q1m_lr2e-5/final`, for the others
`models/grpo2_<size>/final`). All six go through the same conversion, so the size sweep can be shown
side by side.

## 3. Runtime options

Recommendation: start with GGUF files run in an existing llama.cpp app. It needs no app development and
supports our Qwen2 architecture. MLC LLM is the step up if we want phone-GPU speed or our own app.

| Option | Phones | Effort | Speed | Notes |
|---|---|---|---|---|
| GGUF + llama.cpp apps (PocketPal AI, LLM Farm, ChatterUI) | iPhone and Android | Low (~half a day) | Good on CPU | Copy one file to the phone, open it in the app |
| MLC LLM | iPhone and Android | Medium | Best (phone GPU) | Compile the model per phone type; Qwen2 supported |
| Apple Core ML / MLX | iPhone only | Medium to high | Very good on Apple chips | Needs a small iOS app |
| ExecuTorch (PyTorch mobile) | iPhone and Android | High | Good | Most control, most work |

No cloud, server or container is involved in any option: the model runs on the phone.

## 4. Compatibility work

- **Architecture:** our models use the Qwen2 layout (RMSNorm, RoPE, SwiGLU, grouped-query attention,
  QKV bias, tied embeddings), which llama.cpp's converter supports.
- **Tokenizer (main risk):** our own 32k BPE tokenizer (single-digit numbers, reserved logic symbols)
  must convert to GGUF and give identical token ids. Test first: encode 100 GSM8K prompts with the
  original tokenizer and with the GGUF tokenizer and compare.
- **Prompt format:** phone apps assume a chat template; ours is plain completion:
  `Problem: <question>\nSolution:\n`, with the final answer after `####`. Use the app's raw completion
  mode or a custom template that wraps the user's text in that format; stop at the end-of-text token.
- **Quantization accuracy:** 4-bit (Q4_K_M) can cost a few points on small models. Measure GSM8K and
  ProofWriter accuracy of the 16-bit, 8-bit (Q8_0) and 4-bit files on the cluster with llama.cpp before
  choosing; fall back to 8-bit if 4-bit loses more than ~2 points.

## 5. Steps

1. Use the GRPO checkpoints (section 2); convert all six sizes (358M first, as the main demo;
   1M–119M are small and quick; 1.31B checked for memory on the chosen phone).
2. Build llama.cpp on a login or CPU node (CPU only, no GPU needed).
3. Convert the Hugging Face checkpoint to GGUF (`convert_hf_to_gguf.py`), 16-bit.
4. Check the tokenizer: identical token ids on 100 prompts (section 4).
5. Quantize to Q8_0 and Q4_K_M (`llama-quantize`).
6. Measure accuracy of each file on a GSM8K / ProofWriter subset on the cluster; pick the smallest file
   within ~2 points of 16-bit.
7. Copy the file to the laptop (rsync from the login node), then to the phone (AirDrop / USB / Files).
8. Load it in PocketPal AI (or LLM Farm / ChatterUI), set the prompt template and stop token.
9. Test on the phone in airplane mode: 10 problems, record answers, speed (tokens/s) and memory.
10. Optional: MLC LLM build for GPU speed, if time allows.

## 6. Presentation demo

- Phone in airplane mode, screen mirrored to the projector.
- Three prepared problems: a GSM8K word problem, a 6-step arithmetic expression (shows the
  step-by-step rewrite format), a ProofWriter logic question.
- One live problem from the audience.
- Size sweep on the phone: the same 2-step and 6-step problem on the 1M, 10M, 50M and 358M models,
  showing where reasoning appears (the 1M fails, ~50M solves short problems reliably).
- Show the model size (~250 MB) and the speed on screen.
- Fallback: a pre-recorded screen video of the same demo, and the laptop running the same GGUF file.

## 7. Timeline, effort and cost

- GRPO results are in (2026-11-25), so the conversion can start.
- Effort: about half a day for the GGUF route (steps 1–9), plus 1–2 days for MLC LLM if wanted.
- Compute: CPU only on the cluster (conversion, quantization, accuracy checks): a few SU.
- Storage: ~5–6 GB of GGUF files on the cluster (six sizes, 16-bit + 8-bit + 4-bit); ~1.2 GB on the
  phone for all six 4-bit files.

## 8. Risks and open questions

- Which phone (iPhone or Android, model and RAM)? It decides the app and the largest model that fits.
- Tokenizer conversion may need a small fix if our special tokens or digit splitting do not map cleanly.
- Apps' chat templates may need a custom setting for our plain prompt format.
- 4-bit may cost accuracy on the smallest models; measure before choosing.
- Licenses: our weights are trained from scratch; check the training-data licenses (TinyGSM in
  particular) before sharing the model file publicly; a private demo on our own phone is fine.
- The older cluster kernel and libraries could complicate building llama.cpp; building on the laptop is
  the fallback.
