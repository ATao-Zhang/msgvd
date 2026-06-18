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
  --checkpoint path/to/best_model.pt \
  --split test \
  --limit 20 \
  --output results/explain/minimal_sard_20.json
```

Expected outputs:

```text
results/explain/minimal_sard_20.json
results/explain/minimal_metrics_20.csv
results/explain/label_extraction_report_20.json
```

After `--limit 20` succeeds, run `--limit 100`. Do not run the full split until
the small runs have been checked.

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

## Weak Line Labels

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
