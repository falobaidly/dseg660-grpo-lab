"""Result figures from results/*.json -> results/figures/*.png. Run after `grpo_lab.py eval`."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"; F = os.path.join(R, "figures"); os.makedirs(F, exist_ok=True)
c = json.load(open(f"{R}/calibration.json")); t = json.load(open(f"{R}/train_meta.json")); v = json.load(open(f"{R}/eval.json"))
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"

# 1. calibration: group outcome mix + mixed groups per minute
s4, s8 = c["settings"]["4"], c["settings"]["8"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
labels = ["2 x 4", "1 x 8"]
for i, s in enumerate((s4, s8)):
    n = s["groups"]; parts = [s["all_zero_groups"], s["mixed_groups"], s["all_correct_groups"]]
    left = 0
    for p, col, name in zip(parts, (GREY, ORANGE, BLUE), ("all 0 (tied)", "mixed", "all 1 (tied)")):
        ax[0].barh(i, 100 * p / n, left=left, color=col, label=name if i == 0 else None)
        ax[0].text(left + 50 * p / n, i, f"{p}", ha="center", va="center", color="white", fontsize=10)
        left += 100 * p / n
ax[0].set_yticks([0, 1], labels); ax[0].set_xlabel("% of groups"); ax[0].set_title("Calibration: group outcomes (128 cases, 1,024 answers each)")
ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
x = range(2)
ax[1].bar([i - 0.2 for i in x], [s4["mixed_per_min"], s8["mixed_per_min"]], 0.4, color=ORANGE, label="mixed groups / min")
ax[1].bar([i + 0.2 for i in x], [s4["answers_in_mixed_groups_per_min"], s8["answers_in_mixed_groups_per_min"]], 0.4, color=BLUE, label="answers in mixed groups / min")
for i, s in enumerate((s4, s8)):
    ax[1].text(i - 0.2, s["mixed_per_min"], f"{s['mixed_per_min']:.0f}", ha="center", va="bottom")
    ax[1].text(i + 0.2, s["answers_in_mixed_groups_per_min"], f"{s['answers_in_mixed_groups_per_min']:.0f}", ha="center", va="bottom")
ax[1].set_xticks(list(x), labels); ax[1].set_title("Gradient-carrying samples per sampling minute"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{F}/calibration.png", dpi=150); plt.close(fig)

# 2. training curves: reward and fraction of tied groups (rolling mean)
h = [x for x in t["log_history"] if "reward" in x]
def roll(a, k=10): return [sum(a[max(0, i - k + 1):i + 1]) / len(a[max(0, i - k + 1):i + 1]) for i in range(len(a))]
steps = [x["step"] for x in h]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(steps, [x["reward"] for x in h], color=GREY, lw=0.8, label="per step")
ax[0].plot(steps, roll([x["reward"] for x in h]), color=BLUE, lw=2, label="10-step mean")
ax[0].set_title("GRPO training reward (G = %d)" % t["group_size"]); ax[0].set_xlabel("step"); ax[0].set_ylim(-0.02, 1.02); ax[0].legend(frameon=False)
if "frac_reward_zero_std" in h[0]:
    z = [x["frac_reward_zero_std"] for x in h]
    ax[1].plot(steps, roll(z), color=ORANGE, lw=2)
    ax[1].set_title("Fraction of tied groups (zero advantage), 10-step mean"); ax[1].set_xlabel("step"); ax[1].set_ylim(-0.02, 1.02)
fig.tight_layout(); fig.savefig(f"{F}/training.png", dpi=150); plt.close(fig)

# 3. held-out comparison
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, m in enumerate(("base", "adapter")):
    ax[0].bar(i - 0.2, v[m]["scorable"], 0.4, color=GREY, label="scorable" if i == 0 else None)
    ax[0].bar(i + 0.2, v[m]["correct"], 0.4, color=BLUE, label="correct" if i == 0 else None)
    ax[0].text(i - 0.2, v[m]["scorable"], v[m]["scorable"], ha="center", va="bottom")
    ax[0].text(i + 0.2, v[m]["correct"], v[m]["correct"], ha="center", va="bottom")
ax[0].set_xticks([0, 1], ["base", "GRPO adapter"]); ax[0].set_ylim(0, 260); ax[0].axhline(240, color="black", lw=0.5, ls=":")
ax[0].set_title("Held-out (240 cases, greedy, 3-token cap)"); ax[0].legend(frameon=False, loc="lower right")
bars = [v["incorrect_to_correct"], v["correct_to_incorrect"]]
ax[1].bar([0, 1], bars, color=[BLUE, ORANGE])
for i, b in enumerate(bars): ax[1].text(i, b, b, ha="center", va="bottom")
ax[1].set_xticks([0, 1], ["incorrect -> correct", "correct -> incorrect"]); ax[1].set_title("Case-level changes, base -> adapter")
fig.tight_layout(); fig.savefig(f"{F}/heldout.png", dpi=150); plt.close(fig)
print("wrote", os.listdir(F))
