# Results summary (auto-generated from results/*.json)

## Rollout calibration (128 cases, T=1.0, top_p=1.0, top_k=0, max_new_tokens=3)

| setting | groups | answers | scorable | scorable rate | correct | mixed | mixed % | all-1 | all-0 | sampling s | mixed/min | mixed-group answers/min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2x4 | 256 | 1024 | 549 | 0.536 | 342 | 43 | 16.8 | 63 | 150 | 17.4 | 148.2 | 593.0 |
| 1x8 | 128 | 1024 | 558 | 0.545 | 351 | 28 | 21.9 | 29 | 71 | 17.5 | 96.2 | 769.4 |

Recommended group size: **8** (max answers-with-nonzero-advantage per sampling minute).

## Training

- GPU: Tesla T4 (14.56 GB), CUDA 13.0, Python 3.13.16
- Model: Qwen/Qwen3-1.7B @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, seed 3407
- Packages: unsloth==2026.11.12, unsloth_zoo==2026.11.12, torch==2.14.1+cu130, transformers==5.17.0, trl==1.15.0, peft==0.21.2, bitsandbytes==0.50.2, accelerate==1.15.0, datasets==4.8.5
- Group size 8, 1 prompts/step, 200 steps, 200 unique cases
- Wall time (incl. rollouts): 4.0 min; peak allocated VRAM: 2.12 GB
- Mean train reward: 0.578; mean fraction of zero-std (tied) groups: 0.86
- LoRA: {'r': 16, 'lora_alpha': 16, 'lora_dropout': 0.0, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']}
- GRPO config: `{"output_dir": "grpo_out", "seed": 3407, "data_seed": 3407, "learning_rate": 1e-05, "lr_scheduler_type": "cosine", "weight_decay": 0.0, "optim": "adamw_8bit", "max_grad_norm": 0.1, "per_device_train_batch_size": 8, "gradient_accumulation_steps": 1, "num_generations": 8, "max_completion_length": 3, "max_steps": 200, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "beta": 0.0, "loss_type": "grpo", "logging_steps": 1, "save_strategy": "no", "report_to": []}`
- Config fields not supported by installed TRL: ['warmup_ratio', 'max_prompt_length']

## Held-out evaluation (240 cases, greedy, max_new_tokens=3)

| model | scorable | correct |
|---|---|---|
| base | 124/240 | 86/240 |
| adapter | 239/240 | 137/240 |

Incorrect -> correct: **57**; correct -> incorrect: **6**; answer text changed: 228

### Example cases (for the interpretation section)

- `en_8793_5` ref **C** | base 'The most likely' -> adapter 'C\n\n**'
  Options: ['Localized Scleroderma', 'Pleomorphic Lipoma', 'Adamantinoma of the Long Bones']
  Findings: - Muscle atrophy
- Slow growth
- Joint pain
- Local swelling
- Reduced mobility of nearby joint
- Slight fever
- Weight loss
- `en_12272_1` ref **C** | base 'The most likely' -> adapter 'C\n\n**'
  Options: ['Nasal Disease', 'Deficiency-cold cough', 'Sinobronchial Syndrome']
  Findings: - Cough
- Headache
- Nasal congestion
- Runny nose
- `en_23823_2` ref **C** | base 'The most likely' -> adapter 'C'
  Options: ['Progressive Spinal Muscular Atrophy', 'Radial Tunnel Syndrome', 'Median Nerve Injury']
  Findings: - 'Ape hand' deformity
- Neuroma in continuity
- Neurotmesis
- Intraneural fiber damage
- `en_10545_1` ref **B** | base 'The most likely' -> adapter 'B'
  Options: ['Hypopharyngeal Malignant Tumor', 'Vagus Nerve Injury', "Riedel's Thyroiditis"]
  Findings: - Dysphagia
- Mild hoarseness
- Difficulty speaking clearly after prolonged conversation
- Occasional heartburn
- `en_21576_1` ref **C** | base 'The most likely' -> adapter 'C\n\n**'
  Options: ['Allergic Vaginitis', 'Nonspecific Vaginitis', 'Candidal Vulvovaginitis']
  Findings: - Vaginal itching
- Burning sensation upon urination
- Redness around the vulva
- Slight swelling of the vulva
