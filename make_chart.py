# make_chart.py - the README chart: real share of buyers vs the model's predicted share,
# per price band (model = average probability of "Yes", averaged over the 3 training runs).
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

bands = [-0.01, 2, 5, 10, 20, 100]; labels = ["$0–2", "$2–5", "$5–10", "$10–20", "$20+"]
shares = []
for tag in ["main_run", "main_run2", "main_run3"]:
    r = pd.read_csv(f"results/{tag}_500people.csv")
    r["band"] = pd.cut(r.price, bands, labels=labels)
    shares.append(r.groupby("band", observed=True).p_yes.mean() * 100)
real = r.groupby("band", observed=True).truth.mean() * 100
model = pd.concat(shares, axis=1).mean(axis=1)

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

w, gap = 0.36, 0.02
for i, (rv, mv) in enumerate(zip(real, model)):
    for x, v, c in [(i - w/2 - gap/2, rv, BLUE), (i + w/2 + gap/2, mv, ORANGE)]:
        ax.add_patch(FancyBboxPatch((x - w/2, 0), w, v, boxstyle="round,pad=0,rounding_size=0.04",
                                    mutation_aspect=40, facecolor=c, edgecolor="none"))
    ax.text(i - w/2 - gap/2, rv + 1.5, f"{rv:.0f}%", ha="center", color=INK2, fontsize=9.5)
    ax.text(i + w/2 + gap/2, mv + 1.5, f"{mv:.0f}%", ha="center", color=INK2, fontsize=9.5)

ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylim(0, 85)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, color=INK2)
ax.set_yticks([0, 20, 40, 60, 80]); ax.set_yticklabels([f"{v}%" for v in [0, 20, 40, 60, 80]], color=MUTED)
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE); ax.tick_params(length=0)
ax.set_xlabel("Price the person was shown", color=INK2, labelpad=8)
ax.set_ylabel("Share who would buy", color=INK2)
fig.text(0.06, 0.95, "Share of people who would buy, by price: real vs simulated", fontsize=13.5, color=INK, weight="bold")
fig.text(0.06, 0.905, "500 held-out test people, 40 grocery products. Simulated = fine-tuned model's average P(Yes), mean of 3 training runs.",
         fontsize=9.5, color=INK2)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="Real people"), Patch(color=ORANGE, label="Fine-tuned model")],
          frameon=False, loc="upper right", labelcolor=INK2)
fig.subplots_adjust(top=0.84, left=0.09, right=0.98, bottom=0.12)
fig.savefig("figures/real_vs_simulated_by_price.png", facecolor=SURF)
print(pd.DataFrame({"real": real.round(1), "model": model.round(1)}))
