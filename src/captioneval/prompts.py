"""Prompt templates (condensed). Replace with your own for production use."""

CAPTION_TO_FACTS = """You decompose an image caption into atomic facts.
Return ONLY JSON: {"language": "...", "has_text_on_image": bool,
 "facts": [{"id": "f1", "text": "...", "importance": "critical|important|minor",
            "category": "object|attribute|spatial|action|style|camera|mood|text|other",
            "span": "<verbatim fragment of the caption>", "quoted_text": null|"...",
            "hedged": bool}]}
Rules: one claim per fact; keep hedges ("probably"); copy quoted lettering verbatim;
2-5 critical facts (main object / action / key text); do not add anything not in the caption.
Caption:
{caption}"""

IMAGE_TO_FACTS = """Look at the image and list atomic visual facts using the same schema
(object, attribute, spatial, action, style, camera, mood, text, other) with importance
critical|important|minor. Describe only what is visible. Return ONLY JSON."""

JUDGE_FACTS = """Given the image and a list of facts, give each fact a verdict:
full | partial | wrong (for caption facts) / missing | contradicted (for reference facts),
plus a one-sentence reason. Return ONLY JSON: {"verdicts": [{"id": "...", "verdict": "...", "reason": "..."}]}
Facts:
{facts}"""
