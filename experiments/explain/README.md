# Minimal MSAVD Execution-Path Localization

This directory contains the first-stage minimal experiment for vulnerability
statement localization on top of the existing MSAVD/DeepWuKong detector.

The goal is to run a small `--limit 20` pipeline before any full-scale
experiment:

1. read an existing split such as `data/SARD/test.json`;
2. load a trained MSAVD checkpoint;
3. output vulnerability probability for each sample;
4. derive node-level evidence scores;
5. derive path-level scores;
6. map node scores to source line numbers;
7. save Top-1/Top-3/Top-5 localization candidates;
8. write result JSON, metrics CSV, and a label extraction report.

## Run

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json /server/path/to/SARD/test.json \
  --checkpoint /server/path/to/best.ckpt \
  --w2v /server/path/to/w2v.wv \
  --output results/explain/minimal_sard_20.json \
  --limit 20 \
  --label_strategy xfg_stem
```

Expected outputs:

```text
results/explain/minimal_sard_20.json
results/explain/minimal_metrics_20.csv
results/explain/label_extraction_report_20.json
```

After `--limit 20` succeeds, run `--limit 100`. Do not run the full split until
the small runs have been checked.

## Server Run

Activate the same environment used for MSAVD training:

```bash
conda activate <env_name>
```

Check dependencies and path visibility first:

```bash
python experiments/explain/check_explain_env.py \
  --data_json /server/path/to/SARD/test.json \
  --w2v /server/path/to/w2v.wv \
  --checkpoint /server/path/to/best.ckpt \
  --output results/explain/env_check_report.json
```

Then run the minimal localization experiment:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json /server/path/to/SARD/test.json \
  --checkpoint /server/path/to/best.ckpt \
  --w2v /server/path/to/w2v.wv \
  --output results/explain/minimal_sard_20.json \
  --limit 20 \
  --label_strategy xfg_stem
```

If the server does not have `w2v.wv`, pass a saved vocabulary instead:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json /server/path/to/SARD/test.json \
  --checkpoint /server/path/to/best.ckpt \
  --vocab /server/path/to/vocab.pkl \
  --output results/explain/minimal_sard_20.json \
  --limit 20 \
  --label_strategy xfg_stem
```

Path arguments are explicit on purpose. The legacy `--split test` fallback is
kept for local repo-style layouts, but server runs should pass `--data_json`.

Checkpoint loading is strict by default. The runner reads PyTorch Lightning
checkpoints via `ckpt["state_dict"]` when present, adapts common prefixes such as
`model.`, `net.`, and `module.`, prints a key-loading report, and stops if
`loaded_ratio < 0.95`. Only use relaxed loading after inspecting the report:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json /server/path/to/SARD/test.json \
  --checkpoint /server/path/to/best.ckpt \
  --w2v /server/path/to/w2v.wv \
  --output results/explain/minimal_sard_20.json \
  --limit 20 \
  --label_strategy xfg_stem \
  --no_strict_checkpoint
```

## Evidence Scores

This is a minimal version. If the model exposes `forward_with_evidence`, the
script reuses its `node_scores` and `path_scores`. In the current branch,
`node_scores` are attention-style gate scores from the graph encoder and
`path_scores` are the predicted vulnerable probability before line aggregation.

If no explicit evidence is available, the script falls back to a runnable
temporary score:

- node score: L2 norm of node embeddings, or uniform scores if embeddings are
  unavailable;
- path score: predicted vulnerable probability.

The final node score is:

```python
final_node_score = path_score * node_score
```

Line scores use max pooling over nodes on the same source line:

```python
line_score[line_no] = max(node_score of nodes on this line)
```

Later stages should replace this minimal evidence with formal attention,
counterfactual, GNNExplainer, or PGExplainer scores.

## Score Modes

Available `--score_mode` values:

```text
attention
semantic
attention_semantic
counterfactual
attention_semantic_cf
```

The default remains the attention-only baseline:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json data/SARD/test.json \
  --checkpoint ts_logger/DeepWuKong/SARD/version_4/checkpoints/epoch=32-step=10922-val_loss=0.1582.ckpt \
  --w2v data/SARD/w2v.wv \
  --output results/explain/baseline_attention_only_sard_full.json \
  --limit 2648 \
  --label_strategy xfg_stem \
  --score_mode attention
```

Semantic-only ranking:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json data/SARD/test.json \
  --checkpoint ts_logger/DeepWuKong/SARD/version_4/checkpoints/epoch=32-step=10922-val_loss=0.1582.ckpt \
  --w2v data/SARD/w2v.wv \
  --output results/explain/semantic_only_sard_full.json \
  --limit 2648 \
  --label_strategy xfg_stem \
  --score_mode semantic
```

Attention plus semantic risk:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json data/SARD/test.json \
  --checkpoint ts_logger/DeepWuKong/SARD/version_4/checkpoints/epoch=32-step=10922-val_loss=0.1582.ckpt \
  --w2v data/SARD/w2v.wv \
  --output results/explain/attention_semantic_sard_full_w03.json \
  --limit 2648 \
  --label_strategy xfg_stem \
  --score_mode attention_semantic \
  --semantic_weight 0.3
```

Counterfactual-only ranking. Run `--limit 20` first, then `--limit 100`, before
running the full split:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json data/SARD/test.json \
  --checkpoint ts_logger/DeepWuKong/SARD/version_4/checkpoints/epoch=32-step=10922-val_loss=0.1582.ckpt \
  --w2v data/SARD/w2v.wv \
  --output results/explain/counterfactual_only_sard_full.json \
  --limit 2648 \
  --label_strategy xfg_stem \
  --score_mode counterfactual \
  --cf_top_k 10
```

Attention plus semantic plus counterfactual ranking:

```bash
python experiments/explain/run_explain_minimal.py \
  --data_json data/SARD/test.json \
  --checkpoint ts_logger/DeepWuKong/SARD/version_4/checkpoints/epoch=32-step=10922-val_loss=0.1582.ckpt \
  --w2v data/SARD/w2v.wv \
  --output results/explain/attention_semantic_cf_sard_full.json \
  --limit 2648 \
  --label_strategy xfg_stem \
  --score_mode attention_semantic_cf \
  --attention_weight 0.1 \
  --semantic_weight 0.7 \
  --cf_weight 0.2 \
  --cf_top_k 10
```

`attention_score`, `semantic_score`, and `cf_score` are normalized to `[0, 1]`
per sample. For `attention_semantic`, the final score is:

```python
final_score = (1 - semantic_weight) * attention_score_norm + semantic_weight * semantic_score_norm
```

For `attention_semantic_cf`, the three weights are normalized before fusion:

```python
final_score = attention_weight * attention_score_norm + semantic_weight * semantic_score_norm + cf_weight * cf_score_norm
```

## Semantic Risk Score

Semantic Risk Score is a clipped `[0, 1]` rule-based score:

- dangerous copy/string APIs: `+1.0`
  `strcpy`, `wcscpy`, `strcat`, `wcscat`, `sprintf`, `vsprintf`, `gets`,
  `memcpy`, `memmove`, `strncpy`, `wcsncpy`
- memory allocation/free APIs: `+0.7`
  `malloc`, `calloc`, `realloc`, `free`, `alloca`, `new`, `delete`
- input/external source APIs: `+0.6`
  `scanf`, `fscanf`, `sscanf`, `fgets`, `fread`, `read`, `recv`, `getenv`,
  `argv`, `atoi`, `atol`, `strtol`
- array/pointer access: `+0.5`
  `[]`, `->`, and pointer-like assignment
- length/boundary functions: `+0.4`
  `strlen`, `wcslen`, `sizeof`
- arithmetic operators: `+0.2`
  `+`, `-`, `*`, `/`, `%`, `<<`, `>>`

## Counterfactual Score

Counterfactual Score is a lightweight post-hoc perturbation score. It does not
train a new model and does not change preprocessing.

For each sample, the runner first builds an `attention_semantic` candidate order
and only perturbs the first `--cf_top_k` lines. For a candidate line, all graph
nodes whose `Data.line_ids` equal that source line are masked by replacing their
`Data.x` token ids with the vocabulary PAD id. Optional `stmt_features` for the
same nodes are zeroed. The model is then forwarded again:

```text
cf_score(line) = max(0, P_vul(original) - P_vul(mask_line))
```

If the graph does not expose `line_ids` or `x`, or if a candidate line has no
matching node, the runner prints a warning and leaves that line with
`cf_score = 0`, `masked_prob = null`, and `prob_drop = 0`. It does not invent
counterfactual scores.

Each `top_lines` entry includes:

```text
attention_score
semantic_score
cf_score
final_score
semantic_tags
raw_attention_score
masked_prob
prob_drop
```

## Line Labels

For SARD/XFG minimal localization, the default ground truth line strategy is
`xfg_stem`. It uses the source line embedded in the XFG file name:

```text
.../XFG/87595/call/116.xfg.pkl -> true_vul_lines = [116]
```

Negative samples always receive `true_vul_lines = []` and
`label_source = "non_vulnerable"` under this strategy. This prevents broad
comment keywords from creating false line labels for non-vulnerable samples.

Available strategies:

- `xfg_stem`: default; use the first filename segment before `.`, for example
  `47.xfg.pkl -> 47`.
- `keyword_comment`: legacy weak-label keyword/API extraction; useful only for
  manual checks, not the default.
- `hybrid`: try `xfg_stem` first, then strict keyword comments if the filename
  does not contain a usable line number.

The label report contains:

```text
total_samples
positive_samples
negative_samples
positive_with_line_labels
positive_without_line_labels
negative_with_line_labels_should_be_zero
avg_vul_lines_per_positive_sample
label_source_counter
```

If `negative_with_line_labels_should_be_zero > 0`, the runner prints a warning.

## Weak Keyword Labels

`label_extractor.py` implements SARD/Juliet-style weak labels. It first searches
source lines for:

```text
flaw, bad, sink, CWE, POTENTIAL FLAW, FLAW
```

If that fails, it searches dangerous API calls inside `bad` functions:

```text
gets, strcpy, strcat, sprintf, vsprintf, scanf, sscanf, memcpy, memmove, read, recv
```

Failures are recorded in `label_extraction_report_<limit>.json`; samples are not
silently skipped. Localization metrics are calculated only for vulnerable
samples with successfully extracted line labels.
