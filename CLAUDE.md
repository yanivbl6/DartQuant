# DartQuant — project orientation

## What this repo does

Quantization research on small Llama models (primarily Llama-3.2-1B-Instruct). The active method is **quarot**: four rotations (r1, r2, r3, r4) inserted into the model graph to push outliers off the residual stream so it quantizes cleanly. Each rotation has its own knobs and is treated independently — r1 is online pre-attention, r2 is inside MLP gate/up, r3 is inside attention K, r4 is in down_proj; any can be on/off, hadamard or learned. On top of the rotations we run GPTQ at a configurable bit width and quantize activations (static or dynamic).

Note: `dart` (`calibrater/r1_base_qr.py`, `calibrater/r2_base_qr.py`) — the project's namesake learned-rotation method — is now legacy and rarely run. Code is kept; the live path is quarot + GPTQ + activation cal.

## Pipeline entry points

| Stage | File | Purpose |
|---|---|---|
| Cal + rotations + GPTQ produce | `calibrater/calibrate_act_scales.py` | Runs r1/r2 fitting, GPTQ, observes activations, writes `.pt` cal files |
| 3-way cal sweep | `calibrater/multi_calibration.py` | Wraps `calibrate_model.sh` for parallel cal experiments |
| Inference batches | `fake_quant/run_experiments.py` | Drives runfiles (`data/runs/*.ini`) across multiple GPUs |
| Single run | `fake_quant/Script/dart_gptq_wxaykvz.sh` | One-off command for ad-hoc tests |

## Canonical modules — go through these, don't hand-roll

- **`experiment_config.py`** — single source of truth for the quant tag and all artifact paths. Use `build_quant_tag(args, for_gptq_cache=…, for_cal_cache=…, for_fp16_cal_cache=…)`, `resolve_act_scales_path`, `resolve_gptq_checkpoint_dir`, `resolve_gptaq_checkpoint_dir`. Also exposes `apply_set_preset` (e.g. flips `acc_block_k=32 → groupsize` when not explicitly set) — call it after argparse so your tag matches what the cal script wrote.
- **`fake_quant/int_acc_gemm.py`** — integer GEMM with capped accumulator (Triton + reference). Enabled with `--int_gemm --acc_bits N --acc_block_k N`. Theory and the per-group-vs-per-column scale rationale live in `documentation.md`.
- **`fake_quant/gguf_utils.py`** — GGUF dequant + Q/K reverse-permute on load (llama.cpp's `convert.py` permutes Q/K rows; without the reverse, PPL ~1600 instead of ~18).

## Path layout

| Path | Contents |
|---|---|
| `data/runs/*.ini` | Run definitions consumed by `run_experiments.py` — also the canonical place to look for **example runfiles** when constructing a new one |
| `data/cached_results/` | `<tag>.log` (eval), `<tag>_CAL.log` (cal), `<tag>_results.pb` (cached PPL) |
| `data/act_scales/<model>/` | Activation cal `.pt` files (post-GPTQ and `__fp16` variants) |
| `data/gptq_checkpoints/` | GPTQ checkpoints (no asym-aware GPTAQ) |
| `data/gptaq_checkpoints/` | **Separate** dir for GPTAQ checkpoints — dispatched on `args.gptaq` |
| `../quantized_models/` (one level above repo) | GGUF source files |

## ⚠️ TAG MISMATCH — recurring class of bug

Cached cal files, GPTQ checkpoints, and result caches are keyed by a quant tag built via `experiment_config.build_quant_tag(args, for_gptq_cache=…, for_cal_cache=…, for_fp16_cal_cache=…)` — and the booleans matter (cal-time tags strip auto-T2 and rewrite `_hws-` → `_scalewise-`). Hand-rolled tags or wrong booleans cause either a *path miss* or *silent cal contamination*. **Rule:** every consumer goes through `experiment_config.py`'s resolvers; when fresh-cal and cached-cal PPL disagree, diff `Quant tag:` lines in both `_CAL.log`s before chasing algorithmic theories. Full diagnostic playbook in auto-memory (`project_tag_mismatch_pattern.md`).

## Capabilities flagged in passing

1B-model support (untied embeddings, etc.), static activation quant (precomputed scales via cal), PWL activation (ported from hailo-sdk; reference impl at `~/phase2-sdk/.../optimization_flow.py`), int-GEMM (capped accumulator), GGUF load (Q4_K_M tested).

## Etiquette

Be mindful of the time when launching runs — they're slow and the GPUs are shared, so don't kick them off as an afterthought. Check `nvidia-smi` first. When changing environment variables, update the Dockerfile to match.

## Sim versions (`--sim_version N`)

`--sim_version N` (in `experiment_config.py`) appends `_vN` to **all** artifact tags — result cache, post-GPTQ cal cache, FP16 cal cache, **and** GPTQ checkpoint. Purpose: keep results from different code branches from silently sharing caches. GPTQ is included because bug fixes that touch the GPTQ path would otherwise be masked by reusing an old checkpoint across a bump (the saving from sharing GPTQ across versions isn't worth the risk of silent contamination). Bump rules:

- **Bug-fix bump:** when a fix changes behavior of the *current* major (e.g. 74), increment the suffix → 741, 742, … so old and new results don't collide.
- **Major bump (75, 76, 8, …):** only by explicit user request. Do not bump majors on your own.

The active version + commit map lives in [VERSIONS.md](VERSIONS.md) — append a row whenever you bump.

## Pointers

- `documentation.md` — static-vs-dynamic + GEMM-vs-int_gemm theory.
- `VERSIONS.md` — sim_version history (current version is the bottom row).
- `~/.claude/projects/-workspace-DartQuant/memory/MEMORY.md` — auto-loaded cross-session context (past pitfalls, user preferences, GGUF gotchas).
- `.claude/skills/` — operational playbooks (auto-routed by description; no need to enumerate them here).
