# Methodology

## Goal

Independently check how well each image caption matches its image, and estimate the quality of
a whole dataset. Captions are used for pretraining and SFT of image-generation models; the same
methodology applies to both, but **conclusions must not be transferred between the two**.

Two questions are answered separately:

1. Overall caption quality (are the claims true, is the content covered?).
2. Quality of text-on-image (lettering): a stricter sub-problem on the subset of images with text.

## Classes

- `GOOD` – accurate and complete enough to train on.
- `BAD` – contains material errors, hallucinations or omissions.

## Metrics

Both metrics are importance-weighted fact-level scores:

```
score = Σ_i S_i · W_i  /  Σ_i W_i
```

| | Precision (caption → image) | Recall (image → caption) |
|---|---|---|
| Facts come from | the caption | an independent description of the image |
| S_i | 1 supported · 0.5 partial · 0 wrong | 1 covered · 0.5 partial · 0 missing/wrong |
| W_i | critical 5 · important 3 · minor 1 | same |

Example acceptance gates (configurable, see `configs/gates.yaml`): overall P ≥ 0.95, R ≥ 0.80;
text-on-image P = R = 1.0.

## Pipeline

| # | Step | Output |
|---|---|---|
| 1 | LLM splits the caption into atomic facts with importance | ranked fact list |
| 2 | Human verification of the split (periodic) | trusted decomposition |
| 3 | VLM splits the *image* into atomic facts (reference) | ranked reference list |
| 4 | Human verification of the reference facts (periodic) | trusted reference |
| 5 | Fact matching: caption facts vs image, reference facts vs caption | verdict space |
| 6 | LLM-judge analyses WRONG/PARTIAL pairs and assigns error causes | error taxonomy counts |
| 7 | Compute P / R from verified facts | per-caption and dataset metrics |
| 8 | Rewrite check: are critical facts kept in shorter rewrites? | loss per rewrite length |
| 9 | Text-on-image check on the subset with lettering | text P / R |

## Fact-extraction adequacy gates

A caption → facts row is used only if:

1. the model output parsed;
2. ≥ 98% of fact spans are found verbatim in the caption;
3. content-word coverage of the caption ≥ 0.90;
4. the "has text" flag matches the text facts, and every quote is reproduced verbatim;
5. near-duplicate facts ≤ 5%;
6. the main object is covered by a critical fact.

Implemented in `captioneval.facts.validate_caption_facts`.

## Why weight by importance

LLMs over-slice: one idea often becomes 2–3 facts and ~40–45% of facts are minor.
An unweighted recall over all facts has an inflated denominator and is meaninglessly low, so
metrics are weighted and rewrite quality is measured on **critical facts only**.

## Rewrite evaluation

Rewrites are shorter variants of the same caption (medium / short / shortest) used to reduce
overfitting. Primary metric — *recall of critical facts*: for each critical fact of the source
caption, 1 = conveyed, 0.5 = partially (meaning kept, attribute lost), 0 = missing.
Secondary: share of on-image quotes preserved verbatim; ellipsis truncation of quotes.
Rewrite quality is only meaningful when (a) fact extraction and (b) caption-to-image match have
been validated first.

## Issues the evaluation should detect

- **Text on image:** not mentioned · misread/partial · invented · wrong language or transliteration ·
  watermarks/logos described as content.
- **Correctness:** hallucinated objects · wrong attributes (colour, material, age, emotion) ·
  wrong counts · wrong spatial relations · wrong named entities.
- **Completeness:** no main object/action · no background/composition/viewpoint ·
  no style/medium · no mood/lighting.
- **Metadata (from download):** unused · distorted · junk (file names, "free download", personal data).
- **Style/format:** boilerplate phrasing · repetition/padding · mixed languages.
- **Dataset level:** low diversity of lengths, sources, domains, templates.

## Qualitative findings from the original study (numbers intentionally omitted)

- Captions were precise but far less complete: precision was high, recall clearly lower.
- Errors in precision were mostly wrong attributes (colour, material, state), then spatial/count
  errors, then lettering and misidentified objects; pure hallucinations were a minority.
- Composition/camera and technical-quality facts were the least covered.
- Rewrite recall of critical facts dropped monotonically with compression, and the shortest
  rewrites lost most on-image text; short rewrites often dropped quotes or cut them with "…".
- A source-caption defect was found: tokens gluing two alphabets inside one word.
  Rewrites happened to repair it. `captioneval.rewrites.mixed_script_tokens` detects it.
- Fact extraction by the LLM was validated by annotators and was highly accurate.
