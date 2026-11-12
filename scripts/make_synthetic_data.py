"""Generate a fully synthetic evaluation set (no real images, captions or metrics).

Each row mimics the output of the pipeline after the judge step:
caption facts with verdicts (-> precision) and image facts with verdicts (-> recall).
"""
import json
import random
from pathlib import Path

SEED = 7
N = 300
OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "judged_captions.jsonl"

SUBJECTS = ["a red lighthouse", "a ceramic teapot", "a street musician", "a mountain cabin",
            "a vintage bicycle", "a sailing boat", "a market stall", "a snowy pine forest"]
SIGNS = ["OPEN", "CAFE 24", "EXIT", "SALE", "NORTH PIER"]
CATS = ["object", "attribute", "spatial", "action", "style", "camera", "mood", "text"]


def verdicts(rng, n, p_full, p_part):
    out = []
    for _ in range(n):
        x = rng.random()
        out.append("full" if x < p_full else "partial" if x < p_full + p_part else "wrong")
    return out


def make_row(i, rng):
    lang = rng.choice(["en", "ru"])
    subject = rng.choice(SUBJECTS)
    has_text = rng.random() < 0.3
    n_cap = rng.randint(20, 60)
    imps = rng.choices(["critical", "important", "minor"], [0.05, 0.5, 0.45], k=n_cap)
    imps[0] = "critical"
    cap_v = verdicts(rng, n_cap, 0.87, 0.07)
    n_img = rng.randint(20, 60)
    img_imps = rng.choices(["critical", "important", "minor"], [0.06, 0.5, 0.44], k=n_img)
    img_imps[0] = "critical"
    img_v = [rng.choices(["full", "partial", "missing"], [0.50, 0.20, 0.30])[0]
             for _ in range(n_img)]
    return {
        "id": f"syn-{i:04d}",
        "language": lang,
        "domain": rng.choice(["photo", "render", "illustration", "screenshot"]),
        "caption": f"Synthetic placeholder caption about {subject}.",
        "has_text_on_image": has_text,
        "quotes": [rng.choice(SIGNS)] if has_text else [],
        "caption_facts": [{"id": f"c{k}", "category": rng.choice(CATS), "importance": imp,
                           "verdict": v} for k, (imp, v) in enumerate(zip(imps, cap_v))],
        "image_facts": [{"id": f"i{k}", "category": rng.choice(CATS), "importance": imp,
                         "verdict": v} for k, (imp, v) in enumerate(zip(img_imps, img_v))],
        # critical-fact verdicts per rewrite length (for the rewrite study)
        "rewrites": {
            "medium": verdicts_rw(rng, 0.92),
            "short": verdicts_rw(rng, 0.62),
            "shortest": verdicts_rw(rng, 0.40),
        },
        "rewrite_text_kept": {"medium": rng.random() < 0.8, "short": rng.random() < 0.5,
                              "shortest": rng.random() < 0.25} if has_text else None,
    }


def verdicts_rw(rng, keep):
    n = rng.randint(2, 5)
    return [rng.choices(["full", "partial", "missing"],
                        [keep, (1 - keep) * 0.4, (1 - keep) * 0.6])[0] for _ in range(n)]


if __name__ == "__main__":
    rng = random.Random(SEED)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i in range(N):
            f.write(json.dumps(make_row(i, rng), ensure_ascii=False) + "\n")
    print(f"wrote {N} synthetic rows -> {OUT}")
