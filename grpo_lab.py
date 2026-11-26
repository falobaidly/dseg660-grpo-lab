"""DSEG660 GRPO lab: Qwen3-1.7B + Unsloth 4-bit LoRA on synthetic CoD cases.

Stages (run each in a fresh process, in order):
  python grpo_lab.py calib  --data DATA_DIR      # 4 vs 8 rollout calibration on the 128 calibration cases
  python grpo_lab.py train  --data DATA_DIR      # 200 GRPO steps (group size from calib, or --group-size)
  python grpo_lab.py eval   --data DATA_DIR      # base vs adapter, greedy, 240 held-out cases
  python grpo_lab.py report                      # writes results/summary.md from the JSON results
"""
import argparse, csv, glob, json, os, platform, random, re, sys, time
from verifier import extract_choice, reward

MODEL = "Qwen/Qwen3-1.7B"
SEED = 3407
TEMP, TOP_P, TOP_K = 1.0, 1.0, 0      # fixed sampling for calibration AND GRPO rollouts (top_k=0 = off)
MAX_NEW = 3                            # completion cap everywhere
MAX_SEQ = 1024
SEQS_PER_CALL = 64                     # same generate() batch size (in sequences) for both group sizes
TRAIN_BATCH = 8                        # completions per GRPO step -> 8/G prompts per step
STEPS = 200
LORA = dict(r=16, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
OUT, ADAPTER = "results", "lora_adapter"

PROMPT = ("Clinical findings:\n{findings}\n\n"
          "Which diagnosis is most likely?\nA. {A}\nB. {B}\nC. {C}\n\n"
          "Answer with only the letter A, B, or C.")


# ---------------------------------------------------------------- data
def _read(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        if path.endswith(".json"):
            d = json.load(f)
            if isinstance(d, dict):  # {"cases": [...]} or {"ids": [...]}
                d = next(v for v in d.values() if isinstance(v, list))
            return d
        if path.endswith(".csv"):
            return list(csv.DictReader(f))
        return [l.strip() for l in f if l.strip()]  # .txt: one id per line


def _pick(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    raise KeyError(f"none of {keys} in record keys {list(d)}  -> add the right key name in grpo_lab.py")


def _norm(d):
    """Map one raw record to {id, findings, options[3], answer}. Fails loudly on unknown schemas."""
    lower = {k.lower(): v for k, v in d.items()}
    try:
        opts = _pick(lower, ["options", "choices", "candidates"])
    except KeyError:
        opts = [_pick(lower, [x, f"option_{x}", f"option{x}", f"choice_{x}"]) for x in "abc"]
    if isinstance(opts, str):
        opts = json.loads(opts)
    if isinstance(opts, dict):
        opts = [opts.get(x, opts.get(x.lower())) for x in "ABC"]
    opts = [re.sub(r"^\s*[ABC]\s*[\.\):]\s*", "", str(o)).strip() for o in opts]
    assert len(opts) == 3 and all(opts), f"need 3 options, got {opts}"
    ans = _pick(lower, ["answer", "reference", "reference_letter", "label", "gold", "correct", "answer_letter", "target"])
    if isinstance(ans, int) or str(ans).strip().isdigit():
        ans = "ABC"[int(ans)]
    ans = str(ans).strip()
    if ans.upper().rstrip(".") in ("A", "B", "C"):
        ans = ans.upper().rstrip(".")
    else:
        ans = "ABC"[opts.index(ans)]  # answer given as option text
    findings = _pick(lower, ["findings", "clinical_findings", "symptoms", "presentation", "case", "question", "text"])
    if isinstance(findings, list):
        findings = "\n".join(f"- {x}" for x in findings)
    elif isinstance(findings, dict):
        findings = json.dumps(findings, ensure_ascii=False)
    return {"id": str(_pick(lower, ["id", "case_id", "qid", "uid"])), "findings": str(findings).strip(),
            "options": opts, "answer": ans}


def _find(data_dir, word, exclude=None):
    hits = [p for p in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)
            if os.path.isfile(p) and word in os.path.basename(p).lower()
            and (exclude is None or exclude not in os.path.basename(p).lower())
            and p.endswith((".json", ".jsonl", ".csv", ".txt"))]
    if len(hits) != 1:
        sys.exit(f"expected one '{word}' file under {data_dir}, found {hits}; pass --train/--calib/--held")
    return hits[0]


def load_splits(a):
    train = [_norm(r) for r in _read(a.train or _find(a.data, "train", exclude="calib"))]
    held = [_norm(r) for r in _read(a.held or _find(a.data, "held"))]
    ids = [str(r if not isinstance(r, dict) else _pick({k.lower(): v for k, v in r.items()}, ["id", "case_id", "qid", "uid"]))
           for r in _read(a.calib or _find(a.data, "calib"))]
    by_id = {c["id"]: c for c in train}
    calib = [by_id[i] for i in ids]
    assert (len(train), len(calib), len(held)) == (480, 128, 240), (len(train), len(calib), len(held))
    assert not set(by_id) & {c["id"] for c in held}, "held-out overlaps training pool"
    return train, calib, held


# ---------------------------------------------------------------- model helpers
def revision():
    path = os.path.join(OUT, "model_revision.txt")
    if not os.path.exists(path):  # resolve once, then every stage loads the same commit
        from huggingface_hub import model_info
        os.makedirs(OUT, exist_ok=True)
        open(path, "w").write(model_info(MODEL).sha)
    return open(path).read().strip()


def load_model():
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(MODEL, revision=revision(), max_seq_length=MAX_SEQ,
                                                   load_in_4bit=True, dtype=None,
                                                   use_exact_model_name=True)  # else Unsloth swaps in its own pre-quantized repo and drops the revision
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def build_prompt(tok, c):
    msg = PROMPT.format(findings=c["findings"], A=c["options"][0], B=c["options"][1], C=c["options"][2])
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)  # non-thinking mode


def generate(model, tok, prompts, n=1, sample=True):
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kw = dict(do_sample=True, temperature=TEMP, top_p=TOP_P, top_k=TOP_K) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id, **kw)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def env_info():
    import importlib.metadata as md, torch
    pk = {}
    for p in ["unsloth", "unsloth_zoo", "torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate", "datasets"]:
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    return {"python": platform.python_version(), "packages": pk, "gpu": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
            "cuda": torch.version.cuda, "model": MODEL, "model_revision": revision(), "seed": SEED}


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- stage 1: calibration
def calib(a):
    import torch
    from unsloth import FastLanguageModel
    _, cases, _ = load_splits(a)
    model, tok = load_model()
    FastLanguageModel.for_inference(model)
    prompts = [build_prompt(tok, c) for c in cases]
    settings = {}
    for g in (4, 8):
        reps, per_call = 8 // g, SEQS_PER_CALL // g  # 2x4 or 1x8 per case = 1,024 answers either way
        torch.manual_seed(SEED)
        generate(model, tok, prompts[:per_call], g)  # warm-up, not timed
        torch.cuda.synchronize(); t0 = time.perf_counter()
        raw = []
        for r in range(reps):
            for i in range(0, len(prompts), per_call):
                raw.append((r, i, generate(model, tok, prompts[i:i + per_call], g)))
        torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
        groups = []
        for r, i, outs in raw:
            for j, c in enumerate(cases[i:i + per_call]):
                comps = outs[j * g:(j + 1) * g]
                groups.append({"id": c["id"], "rep": r, "ref": c["answer"], "completions": comps,
                               "rewards": [reward(x, c["answer"]) for x in comps]})
        answers = [x for gr in groups for x in gr["completions"]]
        assert len(answers) == 1024, len(answers)
        mixed = sum(0 < sum(gr["rewards"]) < g for gr in groups)
        settings[g] = {
            "group_size": g, "groups": len(groups), "answers": len(answers),
            "scorable": sum(extract_choice(x) is not None for x in answers),
            "correct": int(sum(sum(gr["rewards"]) for gr in groups)),
            "mixed_groups": mixed, "mixed_pct": 100 * mixed / len(groups),
            "all_correct_groups": sum(sum(gr["rewards"]) == g for gr in groups),
            "all_zero_groups": sum(sum(gr["rewards"]) == 0 for gr in groups),
            "sampling_seconds": elapsed, "mixed_per_min": mixed / (elapsed / 60),
            "answers_in_mixed_groups_per_min": mixed * g / (elapsed / 60),
            "cases_with_any_mixed_group": len({gr["id"] for gr in groups if 0 < sum(gr["rewards"]) < g}),
            "group_records": groups,
        }
        settings[g]["scorable_rate"] = settings[g]["scorable"] / 1024
        print({k: v for k, v in settings[g].items() if k != "group_records"})
    best = max(settings, key=lambda g: settings[g]["answers_in_mixed_groups_per_min"])
    dump("calibration.json", {"env": env_info(), "temperature": TEMP, "top_p": TOP_P, "top_k": TOP_K,
                              "max_new_tokens": MAX_NEW, "seqs_per_generate_call": SEQS_PER_CALL,
                              "settings": settings, "recommended_group_size": best,
                              "rule": "max answers-with-nonzero-advantage per sampling minute"})
    print("recommended group size:", best)


# ---------------------------------------------------------------- stage 2: GRPO training
def train(a):
    import torch
    from unsloth import FastLanguageModel  # must be imported before trl
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    G = a.group_size or json.load(open(os.path.join(OUT, "calibration.json")))["recommended_group_size"]
    assert TRAIN_BATCH % G == 0
    random.seed(SEED); torch.manual_seed(SEED)
    pool, _, _ = load_splits(a)
    model, tok = load_model()
    model = FastLanguageModel.get_peft_model(model, **LORA, use_gradient_checkpointing="unsloth", random_state=SEED)

    # Selection: the whole 480-case pool; GRPOTrainer's seeded sampler draws TRAIN_BATCH//G prompts per step.
    ds = Dataset.from_list([{"prompt": build_prompt(tok, c), "answer": c["answer"], "case_id": c["id"]} for c in pool])
    log = []

    def cod_reward(completions, answer, case_id, **kw):
        rs = [reward(c, r) for c, r in zip(completions, answer)]
        log.extend({"case_id": i, "ref": r, "completion": c, "reward": x} for i, r, c, x in zip(case_id, answer, completions, rs))
        return rs

    wanted = dict(output_dir="grpo_out", seed=SEED, data_seed=SEED, learning_rate=1e-5, lr_scheduler_type="cosine",
                  warmup_ratio=0.05, weight_decay=0.0, optim="adamw_8bit", max_grad_norm=0.1,
                  per_device_train_batch_size=TRAIN_BATCH, gradient_accumulation_steps=1, num_generations=G,
                  max_prompt_length=MAX_SEQ - MAX_NEW, max_completion_length=MAX_NEW, max_steps=STEPS,
                  temperature=TEMP, top_p=TOP_P, top_k=TOP_K, beta=0.0, loss_type="grpo",
                  logging_steps=1, save_strategy="no", report_to="none")
    fields = getattr(GRPOConfig, "__dataclass_fields__", {})
    dropped = [k for k in wanted if k not in fields]  # older/newer TRL may lack a field; recorded, not hidden
    cfg = GRPOConfig(**{k: v for k, v in wanted.items() if k in fields})
    trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=[cod_reward], args=cfg, train_dataset=ds)

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()  # wall time includes rollout generation
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()

    model.save_pretrained(ADAPTER); tok.save_pretrained(ADAPTER)
    with open(os.path.join(OUT, "train_rollouts.jsonl"), "w", encoding="utf-8") as f:
        for i, row in enumerate(log):
            f.write(json.dumps({"rollout_batch": i // TRAIN_BATCH + 1, **row}, ensure_ascii=False) + "\n")
    hist = trainer.state.log_history
    zero_std = [h["frac_reward_zero_std"] for h in hist if "frac_reward_zero_std" in h]
    dump("train_meta.json", {
        "env": env_info(), "group_size": G, "prompts_per_step": TRAIN_BATCH // G, "steps_completed": trainer.state.global_step,
        "selection": "all 480 training-pool cases (incl. the 128 calibration cases); GRPOTrainer's seeded "
                     "RepeatSampler draws prompts; every drawn case_id is in train_rollouts.jsonl",
        "unique_cases_used": len({r["case_id"] for r in log}), "lora": LORA,
        "grpo_config_requested": wanted, "grpo_config_fields_dropped": dropped,
        "grpo_config_effective": {k: getattr(cfg, k) for k in wanted if k in fields},
        "train_wall_seconds": wall, "peak_allocated_gb": peak / 2**30,
        "mean_frac_reward_zero_std": sum(zero_std) / len(zero_std) if zero_std else None,
        "mean_train_reward": sum(r["reward"] for r in log) / len(log),
        "log_history": hist})
    print(f"done: {trainer.state.global_step} steps, {wall/60:.1f} min, peak {peak/2**30:.2f} GB")


# ---------------------------------------------------------------- stage 3: held-out evaluation
def evaluate(a):
    from unsloth import FastLanguageModel
    from peft import PeftModel
    _, _, held = load_splits(a)
    model, tok = load_model()
    model = PeftModel.from_pretrained(model, ADAPTER)
    FastLanguageModel.for_inference(model)
    prompts = [build_prompt(tok, c) for c in held]

    def run():  # greedy, same prompts, same batching, same cap for both models
        return [o for i in range(0, len(prompts), 32) for o in generate(model, tok, prompts[i:i + 32], sample=False)]

    with model.disable_adapter():
        base = run()
    tuned = run()
    rows = [{"id": c["id"], "ref": c["answer"], "findings": c["findings"], "options": c["options"],
             "base": b, "base_choice": extract_choice(b), "base_reward": reward(b, c["answer"]),
             "adapter": t, "adapter_choice": extract_choice(t), "adapter_reward": reward(t, c["answer"])}
            for c, b, t in zip(held, base, tuned)]
    res = {"n": len(rows)}
    for m in ("base", "adapter"):
        res[m] = {"scorable": sum(r[f"{m}_choice"] is not None for r in rows),
                  "correct": int(sum(r[f"{m}_reward"] for r in rows))}
    res["incorrect_to_correct"] = sum(r["base_reward"] == 0 and r["adapter_reward"] == 1 for r in rows)
    res["correct_to_incorrect"] = sum(r["base_reward"] == 1 and r["adapter_reward"] == 0 for r in rows)
    res["changed_answer"] = sum(r["base"] != r["adapter"] for r in rows)
    res["env"] = env_info()
    res["rows"] = rows
    dump("eval.json", res)
    print({k: v for k, v in res.items() if k not in ("rows", "env")})


# ---------------------------------------------------------------- stage 4: summary tables
def report(a):
    L = ["# Results summary (auto-generated from results/*.json)\n"]
    p = lambda n: os.path.join(OUT, n)
    if os.path.exists(p("calibration.json")):
        c = json.load(open(p("calibration.json")))
        L += [f"## Rollout calibration (128 cases, T={c['temperature']}, top_p={c['top_p']}, top_k={c['top_k']}, "
              f"max_new_tokens={c['max_new_tokens']})\n",
              "| setting | groups | answers | scorable | scorable rate | correct | mixed | mixed % | all-1 | all-0 | sampling s | mixed/min | mixed-group answers/min |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, s in c["settings"].items():
            L.append(f"| {s['groups']//128}x{g} | {s['groups']} | {s['answers']} | {s['scorable']} | {s['scorable_rate']:.3f} | "
                     f"{s['correct']} | {s['mixed_groups']} | {s['mixed_pct']:.1f} | {s['all_correct_groups']} | "
                     f"{s['all_zero_groups']} | {s['sampling_seconds']:.1f} | {s['mixed_per_min']:.1f} | "
                     f"{s['answers_in_mixed_groups_per_min']:.1f} |")
        L.append(f"\nRecommended group size: **{c['recommended_group_size']}** ({c['rule']}).\n")
    if os.path.exists(p("train_meta.json")):
        t = json.load(open(p("train_meta.json")))
        e = t["env"]
        L += ["## Training\n", f"- GPU: {e['gpu']} ({e['gpu_total_gb']} GB), CUDA {e['cuda']}, Python {e['python']}",
              f"- Model: {e['model']} @ `{e['model_revision']}`, seed {e['seed']}",
              f"- Packages: {', '.join(f'{k}=={v}' for k, v in e['packages'].items())}",
              f"- Group size {t['group_size']}, {t['prompts_per_step']} prompts/step, {t['steps_completed']} steps, "
              f"{t['unique_cases_used']} unique cases",
              f"- Wall time (incl. rollouts): {t['train_wall_seconds']/60:.1f} min; peak allocated VRAM: {t['peak_allocated_gb']:.2f} GB",
              f"- Mean train reward: {t['mean_train_reward']:.3f}; mean fraction of zero-std (tied) groups: {t['mean_frac_reward_zero_std']}",
              f"- LoRA: {t['lora']}", f"- GRPO config: `{json.dumps(t['grpo_config_effective'])}`",
              f"- Config fields not supported by installed TRL: {t['grpo_config_fields_dropped']}\n"]
    if os.path.exists(p("eval.json")):
        v = json.load(open(p("eval.json")))
        L += ["## Held-out evaluation (240 cases, greedy, max_new_tokens=3)\n", "| model | scorable | correct |", "|---|---|---|",
              f"| base | {v['base']['scorable']}/240 | {v['base']['correct']}/240 |",
              f"| adapter | {v['adapter']['scorable']}/240 | {v['adapter']['correct']}/240 |",
              f"\nIncorrect -> correct: **{v['incorrect_to_correct']}**; correct -> incorrect: **{v['correct_to_incorrect']}**; "
              f"answer text changed: {v['changed_answer']}\n"]
        flips = [r for r in v["rows"] if r["base_reward"] != r["adapter_reward"]] or v["rows"][:1]
        L.append("### Example cases (for the interpretation section)\n")
        for r in flips[:5]:
            L.append(f"- `{r['id']}` ref **{r['ref']}** | base {r['base']!r} -> adapter {r['adapter']!r}\n  "
                     f"Options: {r['options']}\n  Findings: {r['findings'][:300]}")
    open(p("summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calib", "train", "eval", "report"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--train"); ap.add_argument("--calib"); ap.add_argument("--held")
    ap.add_argument("--group-size", type=int)
    a = ap.parse_args()
    {"calib": calib, "train": train, "eval": evaluate, "report": report}[a.stage](a)
