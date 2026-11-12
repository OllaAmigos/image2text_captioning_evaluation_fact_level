"""Compute metrics, gates and rewrite stats on the synthetic set; draw the demo figure."""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from captioneval import caption_precision, image_recall, f1, is_good, bootstrap_ci
from captioneval.metrics import has_critical_error
from captioneval.rewrites import rewrite_critical_recall

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / "data/synthetic/judged_captions.jsonl").open(encoding="utf-8")]
cfg = {"p_gate": 0.95, "r_gate": 0.80}  # see configs/gates.yaml

P, R, good = [], [], []
for r in rows:
    cv = [(f["verdict"], f["importance"]) for f in r["caption_facts"]]
    iv = [(f["verdict"], f["importance"]) for f in r["image_facts"]]
    p, rc = caption_precision(cv), image_recall(iv)
    r["P"], r["R"] = p, rc
    r["good"] = is_good(p, rc, has_critical_error(cv))
    P.append(p); R.append(rc); good.append(r["good"])

lines = ["# Demo results (SYNTHETIC data)", ""]
for name, v in [("Precision", P), ("Recall", R)]:
    lo, hi = bootstrap_ci(v)
    lines.append(f"- {name}: mean {st.fmean(v):.3f}, 95% CI [{lo:.3f}; {hi:.3f}], median {st.median(v):.3f}")
lines.append(f"- GOOD share: {sum(good)/len(good):.1%}")
lines.append(f"- Share of captions passing the acceptance gates "
             f"(P>={cfg['p_gate']}, R>={cfg['r_gate']}): "
             f"{sum(p>=cfg['p_gate'] and rc>=cfg['r_gate'] for p, rc in zip(P, R))/len(P):.1%}")
lines += ["", "## Rewrites: recall of critical facts", "", "| length | recall | text kept |", "|---|---|---|"]
for k in ("medium", "short", "shortest"):
    rec = st.fmean(rewrite_critical_recall(r["rewrites"][k]) for r in rows)
    kept = [r["rewrite_text_kept"][k] for r in rows if r["rewrite_text_kept"]]
    lines.append(f"| {k} | {rec:.2f} | {sum(kept)/len(kept):.0%} |")
(ROOT / "docs" / "demo_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = ["#1f8a5b" if g else "#c93c47" for g in good]
ax[0].scatter(P, R, c=cols, s=14, alpha=.7)
ax[0].axvline(0.85, ls="--", c="gray"); ax[0].axhline(0.5, ls="--", c="gray")
ax[0].set(xlabel="Precision", ylabel="Recall", title="Per-caption P vs R (green = GOOD)")
ax[1].hist(P, bins=25, alpha=.6, label="Precision"); ax[1].hist(R, bins=25, alpha=.6, label="Recall")
ax[1].legend(); ax[1].set(title="Distributions (synthetic data)", xlabel="score")
fig.tight_layout(); fig.savefig(ROOT / "docs/img/demo_scatter.png", dpi=130)
