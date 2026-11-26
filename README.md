# DSEG660 GRPO Lab — Reward Design with CoD

**Report: [REPORT.md](REPORT.md)** (calibration, training measurements, held-out comparison, interpretation, figures).

Qwen/Qwen3-1.7B (non-thinking mode) + Unsloth 4-bit LoRA + TRL GRPO on synthetic Chain-of-Diagnosis cases.

## Files
| file | purpose |
|---|---|
| `verifier.py` | choice extraction + binary reward (pure Python) |
| `test_verifier.py` | correct / incorrect / malformed / ambiguous tests → `results/verifier_tests.txt` |
| `grpo_lab.py` | stages `calib`, `train`, `eval`, `report` |
| `make_figures.py` | result figures → `results/figures/*.png` |
| `grpo_colab.ipynb` | Colab runner for the stages above |
| `lora_adapter/` | saved LoRA adapter + `adapter_config.json` |
| `results/` | `calibration.json`, `train_meta.json`, `train_rollouts.jsonl`, `eval.json`, `summary.md`, `model_revision.txt`, `pip_freeze.txt` |
| `REPORT.md` | report |

## Setup and run (Google Colab, T4)
1. Runtime → T4 GPU. Open `grpo_colab.ipynb`.
2. Upload this folder as `dseg660-grpo-lab.zip` plus `dseg660-cod-data.zip`. Then run all cells.

Or run the stages yourself on any single 16 GB GPU:
```bash
pip install unsloth
python test_verifier.py
python grpo_lab.py calib --data data      # data = unzipped dseg660-cod-data.zip
python grpo_lab.py train --data data      # optional: --group-size 4|8
python grpo_lab.py eval  --data data
python grpo_lab.py report && python make_figures.py
```
The data files are found by name (`*train*`, `*calib*`, `*held*`). If the names differ, pass `--train/--calib/--held PATH`.
The loader accepts JSON, JSONL, or CSV and common field names. An unknown schema stops with an error that lists the record's keys.

## Fixed settings (reproducibility)
- Model `Qwen/Qwen3-1.7B` at a pinned commit. The SHA is resolved once and saved in `results/model_revision.txt`, and every stage reuses it.
- Seed 3407. Prompt template in `grpo_lab.PROMPT`, built with `enable_thinking=False`. Left padding.
- Sampling for calibration and GRPO rollouts: temperature 1.0, top_p 1.0, top_k off, max 3 new tokens.
- Evaluation: greedy decoding, max 3 new tokens. The base model is the same loaded weights with the adapter disabled.
- LoRA r=16, alpha=16, dropout 0, all attention and MLP projections. GRPO: 200 steps, lr 1e-5 cosine (warmup_ratio not supported by TRL 1.15, so no warmup), adamw_8bit, 8 completions per step, beta=0.
- Package versions, the GPU, wall time, and peak allocated VRAM are written to `results/*.json` at run time.

## Data and license
The course data (`dseg660-cod-data.zip`) is not included in this repository; unzip it into `data/` to reproduce. `results/eval.json` and the rollout logs quote case text from those files. The cases are synthetic examples derived from [FreedomIntelligence/CoD-PatientSymDisease](https://huggingface.co/datasets/FreedomIntelligence/CoD-PatientSymDisease) (Apache-2.0).
