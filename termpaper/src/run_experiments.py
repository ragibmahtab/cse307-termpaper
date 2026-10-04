import sys, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paging import gen_trace, fifo, lru, opt, learned, collect, fit

N, PAGES, FRAMES, D = 20000, 200, 20, 40
SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
OUT = "../results"

rows, roll = [], {}
t0 = time.time()
for s in SEEDS:
    trace, shift = gen_trace(N, PAGES, seed=s)
    ttrace, _ = gen_trace(N // 2, PAGES, seed=1000 + s, phase="p1")
    X, y = collect(ttrace, FRAMES, D)
    model = fit(X, y)
    res = {
        "FIFO": fifo(trace, FRAMES),
        "LRU": lru(trace, FRAMES),
        "OPT": opt(trace, FRAMES),
        "Learned-static": learned(trace, FRAMES, model, D),
        "Learned-online": learned(trace, FRAMES, model, D, online=True),
    }
    for name, h in res.items():
        for phase, sl in (("before", slice(0, shift)), ("after", slice(shift, N)), ("overall", slice(0, N))):
            hh = h[sl]
            rows.append(dict(seed=s, policy=name, phase=phase, refs=len(hh),
                             faults=int((~hh).sum()), hit_ratio=hh.mean()))
        if s == 0:
            roll[name] = pd.Series(h.astype(float)).rolling(500).mean().values
    print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/raw_results.csv", index=False)

order = ["FIFO", "LRU", "Learned-static", "Learned-online", "OPT"]
g = df.groupby(["policy", "phase"]).agg(hit_mean=("hit_ratio", "mean"), hit_std=("hit_ratio", "std"),
                                         faults_mean=("faults", "mean"), faults_std=("faults", "std")).reset_index()
g.to_csv(f"{OUT}/summary.csv", index=False)

piv = g.pivot(index="policy", columns="phase", values="hit_mean").loc[order]
piv["drop_abs"] = piv["before"] - piv["after"]
piv["drop_rel_%"] = 100 * piv["drop_abs"] / piv["before"]
piv.round(4).to_csv(f"{OUT}/degradation_table.csv")
fp = g.pivot(index="policy", columns="phase", values="faults_mean").loc[order].round(1)
fp.to_csv(f"{OUT}/faults_table.csv")
print("\nHit ratio (mean over seeds)\n", piv.round(4).to_string())
print("\nPage faults (mean over seeds)\n", fp.to_string())

fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.38; x = np.arange(len(order))
for k, (ph, c) in enumerate((("before", "#4c78a8"), ("after", "#e45756"))):
    sub = g[g.phase == ph].set_index("policy").loc[order]
    ax.bar(x + (k - .5) * w, sub.hit_mean, w, yerr=sub.hit_std, label=f"{ph} shift", color=c, capsize=2)
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=8); ax.set_ylabel("Hit ratio"); ax.legend()
ax.set_title(f"Hit ratio before vs after shift ({len(SEEDS)} seeds, {FRAMES} frames)", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_hit_ratio_phases.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.6))
for name in order:
    ax.plot(roll[name], label=name, lw=1.1)
ax.axvline(N // 2, color="k", ls="--", lw=1); ax.text(N // 2 + 150, 0.97, "shift", fontsize=8)
ax.set_xlabel("reference index"); ax.set_ylabel("rolling hit ratio (500 refs)"); ax.legend(fontsize=7, ncol=3)
ax.set_title("Hit ratio over time, seed 0", fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_rolling_hit_ratio.png", dpi=200); plt.close()
print("done")
