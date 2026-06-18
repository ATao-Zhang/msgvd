import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.explain.counterfactual_scorer import compute_counterfactual_scores  # noqa: E402
from experiments.explain.label_extractor import (  # noqa: E402
    LabelExtractionReport,
    extract_vulnerable_lines,
    graph_file_path,
    read_source_lines,
    source_line_map,
)
from experiments.explain.localization_metrics import calculate_localization_metrics  # noqa: E402
from experiments.explain.semantic_scorer import combine_scores, normalize_scores, semantic_risk_score  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MSAVD execution-path evidence localization.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained MSAVD checkpoint.")
    parser.add_argument("--config", default="configs/dwk.yaml", help="Path to the YAML config.")
    parser.add_argument("--data_json", default=None,
                        help="Explicit split JSON path, e.g. /server/path/to/SARD/test.json.")
    parser.add_argument("--w2v", default=None,
                        help="Explicit word2vec .wv path. Takes precedence over --vocab.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of samples to run.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--vocab", default=None, help="Optional pickled vocabulary path.")
    parser.add_argument("--strict_checkpoint", dest="strict_checkpoint", action="store_true", default=True,
                        help="Require exact checkpoint/model key match after prefix adaptation. Default: enabled.")
    parser.add_argument("--no_strict_checkpoint", dest="strict_checkpoint", action="store_false",
                        help="Allow relaxed loading after prefix adaptation if loaded_ratio >= 0.95.")
    parser.add_argument("--label_strategy", default="xfg_stem",
                        choices=("xfg_stem", "keyword_comment", "hybrid"),
                        help="Line-label extraction strategy. Default: xfg_stem.")
    parser.add_argument("--score_mode", default="attention",
                        choices=(
                            "attention",
                            "semantic",
                            "attention_semantic",
                            "counterfactual",
                            "attention_semantic_cf",
                        ),
                        help="Line ranking score mode. Default keeps the attention-only baseline.")
    parser.add_argument("--semantic_weight", type=float, default=None,
                        help="Semantic weight. Defaults to 0.3, or 0.7 for attention_semantic_cf.")
    parser.add_argument("--cf_top_k", type=int, default=10,
                        help="Only compute counterfactual scores for the current top-K candidate lines. Default: 10.")
    parser.add_argument("--attention_weight", type=float, default=0.1,
                        help="Attention weight for attention_semantic_cf mode. Default: 0.1.")
    parser.add_argument("--cf_weight", type=float, default=0.2,
                        help="Counterfactual weight for attention_semantic_cf mode. Default: 0.2.")
    parser.add_argument("--prefer-explicit-labels", action="store_true",
                        help="Prefer graph metadata line labels before weak SARD/Juliet extraction.")
    return parser.parse_args()


def resolve_semantic_weight(score_mode: str, semantic_weight) -> float:
    if semantic_weight is not None:
        return float(semantic_weight)
    if score_mode == "attention_semantic_cf":
        return 0.7
    return 0.3


def _sample_path(sample) -> str:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        for key in ("xfg_path", "graph_path", "gpickle_path", "path", "file_path"):
            if sample.get(key):
                return str(sample[key])
    raise ValueError(f"Unsupported sample entry in data_json: {sample!r}")


def _resolve_sample_path(path_value: str, data_json_path: Path) -> str:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return str(path)
    candidates = [
        Path.cwd() / path_value,
        data_json_path.parent / path_value,
        ROOT / path_value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return path_value


def load_split_paths(config, split: str, limit: int, data_json: str = None) -> List[str]:
    split_path = Path(data_json) if data_json else ROOT / config.data_folder / config.dataset.name / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    samples = json.loads(split_path.read_text(encoding="utf-8"))
    paths = [_resolve_sample_path(_sample_path(sample), split_path) for sample in samples]
    if limit is not None and limit > 0:
        paths = paths[:limit]
    return paths


def load_vocabulary(config, w2v_path: str = None, vocab_path: str = None):
    from src.vocabulary import Vocabulary

    if w2v_path:
        config.gnn.w2v_path = w2v_path
        return Vocabulary.build_from_w2v(w2v_path)
    if vocab_path:
        return Vocabulary.load_vocabulary(vocab_path)
    raise ValueError("Either --w2v or --vocab must be provided for server execution.")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _strip_prefix(key: str, prefixes: Tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _checkpoint_key_variants(key: str) -> List[str]:
    prefixes = ("model.", "net.", "module.", "_forward_module.", "model.net.", "model.module.")
    stripped = _strip_prefix(key, prefixes)
    variants = [key, stripped]
    for prefix in prefixes:
        if key.startswith(prefix):
            variants.append(key[len(prefix):])
        if stripped.startswith(prefix):
            variants.append(stripped[len(prefix):])
    return list(dict.fromkeys(variants))


def _select_checkpoint_keys(model_state: Dict, checkpoint_state: Dict) -> Tuple[Dict, Dict]:
    adapted = {}
    skipped_shape = {}
    model_keys = set(model_state.keys())
    for ckpt_key, value in checkpoint_state.items():
        matched_key = None
        for candidate_key in _checkpoint_key_variants(ckpt_key):
            if candidate_key in model_keys:
                matched_key = candidate_key
                break
        if matched_key is None:
            continue
        if tuple(model_state[matched_key].shape) != tuple(value.shape):
            skipped_shape[matched_key] = {
                "checkpoint_key": ckpt_key,
                "model_shape": list(model_state[matched_key].shape),
                "checkpoint_shape": list(value.shape),
            }
            continue
        adapted[matched_key] = value
    return adapted, skipped_shape


def _checkpoint_report(checkpoint_path: str, model_state: Dict, checkpoint_state: Dict,
                       adapted_state: Dict, skipped_shape: Dict) -> Dict:
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint_state.keys())
    loaded_keys = set(adapted_state.keys())
    missing = sorted(model_keys - loaded_keys)
    matched_checkpoint_keys = set()
    for ckpt_key in checkpoint_keys:
        if any(candidate in loaded_keys for candidate in _checkpoint_key_variants(ckpt_key)):
            matched_checkpoint_keys.add(ckpt_key)
    unexpected = sorted(checkpoint_keys - matched_checkpoint_keys)
    loaded_ratio = len(loaded_keys) / max(len(model_keys), 1)
    return {
        "checkpoint_path": checkpoint_path,
        "num_model_keys": len(model_keys),
        "num_checkpoint_keys": len(checkpoint_keys),
        "num_loaded_keys": len(loaded_keys),
        "num_missing_keys": len(missing),
        "num_unexpected_keys": len(unexpected),
        "num_shape_mismatch_keys": len(skipped_shape),
        "loaded_ratio": loaded_ratio,
        "first_10_missing_keys": missing[:10],
        "first_10_unexpected_keys": unexpected[:10],
        "first_10_shape_mismatch_keys": list(skipped_shape.items())[:10],
        "first_30_model_keys": sorted(model_keys)[:30],
        "first_30_checkpoint_keys": sorted(checkpoint_keys)[:30],
    }


def _print_checkpoint_report(report: Dict):
    print("[checkpoint_report]")
    for key in (
        "checkpoint_path",
        "num_model_keys",
        "num_checkpoint_keys",
        "num_loaded_keys",
        "num_missing_keys",
        "num_unexpected_keys",
        "num_shape_mismatch_keys",
        "loaded_ratio",
        "first_10_missing_keys",
        "first_10_unexpected_keys",
        "first_10_shape_mismatch_keys",
    ):
        print(f"{key}: {report[key]}")


def build_model(config, vocab, checkpoint_path: str, device, strict_checkpoint: bool = True):
    from src.models.vd import DeepWuKong

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model = DeepWuKong(config, vocab, vocab.get_vocab_size(), vocab.get_pad_id())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint state_dict type: {type(state_dict)!r}")

    model_state = model.state_dict()
    adapted_state, skipped_shape = _select_checkpoint_keys(model_state, state_dict)
    report = _checkpoint_report(checkpoint_path, model_state, state_dict, adapted_state, skipped_shape)
    _print_checkpoint_report(report)

    if report["loaded_ratio"] < 0.95:
        raise RuntimeError(
            "Checkpoint/model key match is too low. "
            f"loaded_ratio={report['loaded_ratio']:.4f}. "
            f"First 30 model keys: {report['first_30_model_keys']}. "
            f"First 30 checkpoint keys: {report['first_30_checkpoint_keys']}."
        )

    if strict_checkpoint and (
        report["num_missing_keys"] > 0
        or report["num_unexpected_keys"] > 0
        or report["num_shape_mismatch_keys"] > 0
    ):
        raise RuntimeError(
            "Strict checkpoint loading failed after prefix adaptation. "
            "Use --no_strict_checkpoint only after confirming the reported keys are acceptable."
        )

    load_result = model.load_state_dict(adapted_state, strict=strict_checkpoint)
    if not strict_checkpoint:
        missing = list(load_result.missing_keys)
        unexpected = list(load_result.unexpected_keys)
        if missing or unexpected:
            print(f"[warn] relaxed checkpoint loading missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _normalize_scores(scores):
    scores = scores.detach().float().cpu()
    if scores.numel() == 0:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if torch.isclose(max_score, min_score):
        return torch.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _evidence_scores(model, graph_batch) -> Tuple[object, int, Dict[str, object]]:
    if hasattr(model, "forward_with_evidence"):
        evidence = model.forward_with_evidence(graph_batch)
        logits = evidence["logits"]
    else:
        evidence = {}
        logits = model(graph_batch)
    prob = torch.softmax(logits, dim=-1)[0, 1]
    pred_label = int(torch.argmax(logits, dim=-1)[0].item())

    node_scores = evidence.get("node_scores")
    if node_scores is None:
        node_embeddings = evidence.get("node_embeddings")
        if node_embeddings is not None:
            node_scores = torch.linalg.vector_norm(node_embeddings, ord=2, dim=-1)
        else:
            node_scores = torch.ones(graph_batch.num_nodes, device=logits.device)
    node_scores = _normalize_scores(node_scores)

    path_scores = evidence.get("path_scores")
    if path_scores is not None and torch.numel(path_scores) > 0:
        path_score = float(path_scores.detach().flatten()[0].cpu())
        node_scores = node_scores * path_score
    else:
        path_score = float(prob.detach().cpu())
    evidence["minimal_path_score"] = torch.tensor(path_score)
    return node_scores, pred_label, {"logits": logits.detach().cpu(), "prob": prob.detach().cpu(), **evidence}


def _normalize_weight_triplet(attention_weight: float, semantic_weight: float, cf_weight: float) -> Tuple[float, float, float]:
    weights = [max(0.0, float(attention_weight)), max(0.0, float(semantic_weight)), max(0.0, float(cf_weight))]
    total = sum(weights)
    if total <= 0:
        return 0.1, 0.7, 0.2
    return weights[0] / total, weights[1] / total, weights[2] / total


def _combine_line_score(attn_score: float, sem_score: float, cf_score: float, score_mode: str,
                        semantic_weight: float, attention_weight: float, cf_weight: float) -> float:
    if score_mode in ("attention", "semantic", "attention_semantic"):
        return combine_scores(attn_score, sem_score, score_mode, semantic_weight)
    if score_mode == "counterfactual":
        return cf_score
    if score_mode == "attention_semantic_cf":
        attn_w, sem_w, cf_w = _normalize_weight_triplet(attention_weight, semantic_weight, cf_weight)
        return attn_w * attn_score + sem_w * sem_score + cf_w * cf_score
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def line_rankings(node_scores, line_ids, code_by_line: Dict[int, str],
                  score_mode: str = "attention", semantic_weight: float = 0.3,
                  cf_scores: Dict[int, Dict] = None, attention_weight: float = 0.1,
                  cf_weight: float = 0.2) -> List[Dict]:
    line_scores: Dict[int, float] = {}
    for score, line_no in zip(node_scores.tolist(), line_ids.detach().cpu().tolist()):
        line_no = int(line_no)
        line_scores[line_no] = max(line_scores.get(line_no, float("-inf")), float(score))

    line_items = sorted(line_scores.items(), key=lambda item: item[0])
    attention_raw = [float(score) for _, score in line_items]
    attention_norm = normalize_scores(attention_raw)
    semantic_infos = [semantic_risk_score(code_by_line.get(int(line_no), "")) for line_no, _ in line_items]
    semantic_norm = normalize_scores([float(info["semantic_score"]) for info in semantic_infos])
    cf_scores = cf_scores or {}
    cf_raw = [float(cf_scores.get(int(line_no), {}).get("cf_score", 0.0)) for line_no, _ in line_items]
    cf_norm = normalize_scores(cf_raw)

    rows = []
    for (line_no, raw_attention), attn_score, sem_score, sem_info, cf_score in zip(
        line_items,
        attention_norm,
        semantic_norm,
        semantic_infos,
        cf_norm,
    ):
        cf_info = cf_scores.get(int(line_no), {})
        final_score = _combine_line_score(
            attn_score,
            sem_score,
            cf_score,
            score_mode,
            semantic_weight,
            attention_weight,
            cf_weight,
        )
        rows.append({
            "line": int(line_no),
            "score": float(final_score),
            "attention_score": float(attn_score),
            "semantic_score": float(sem_score),
            "cf_score": float(cf_score),
            "final_score": float(final_score),
            "semantic_tags": sem_info["semantic_tags"],
            "code": code_by_line.get(int(line_no), ""),
            "raw_attention_score": float(raw_attention),
            "masked_prob": cf_info.get("masked_prob"),
            "prob_drop": float(cf_info.get("prob_drop", 0.0)),
        })
    return sorted(rows, key=lambda item: (-item["final_score"], item["line"]))


def _needs_counterfactual(score_mode: str) -> bool:
    return score_mode in ("counterfactual", "attention_semantic_cf")


def _candidate_mode_for_counterfactual(score_mode: str) -> str:
    if score_mode in ("counterfactual", "attention_semantic_cf"):
        return "attention_semantic"
    return score_mode


def explain_one(model, config, vocab, xfg_path: str, device, prefer_explicit_labels: bool,
                label_strategy: str, score_mode: str, semantic_weight: float,
                cf_top_k: int, attention_weight: float, cf_weight: float) -> Dict:
    import networkx as nx
    from torch_geometric.data import Batch

    from src.datas.graphs import XFG

    graph_nx = nx.read_gpickle(xfg_path)
    label = int(graph_nx.graph["label"])
    source_lines = read_source_lines(graph_file_path(graph_nx))
    code_by_line = source_line_map(graph_nx, source_lines)
    true_lines, label_source = extract_vulnerable_lines(
        graph_nx,
        xfg_path=xfg_path,
        label=label,
        strategy=label_strategy,
        source_lines=source_lines,
        prefer_explicit=prefer_explicit_labels,
    )

    xfg = XFG(xfg=graph_nx)
    data = xfg.to_torch(vocab, config.dataset.token.max_parts)
    graph_batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        node_scores, pred_label, evidence = _evidence_scores(model, graph_batch)

    cf_scores = {}
    cf_warnings: List[str] = []
    if _needs_counterfactual(score_mode):
        preliminary = line_rankings(
            node_scores,
            data.line_ids,
            code_by_line,
            score_mode=_candidate_mode_for_counterfactual(score_mode),
            semantic_weight=semantic_weight,
        )
        candidate_lines = [item["line"] for item in preliminary[:max(0, int(cf_top_k))]]
        cf_scores, cf_warnings = compute_counterfactual_scores(
            model=model,
            data=data,
            candidate_lines=candidate_lines,
            original_prob=float(evidence["prob"]),
            device=device,
            pad_id=vocab.get_pad_id(),
            cf_top_k=cf_top_k,
        )
        for warning in cf_warnings[:3]:
            print(f"[warning] counterfactual fallback for {xfg_path}: {warning}")

    ranked = line_rankings(
        node_scores,
        data.line_ids,
        code_by_line,
        score_mode=score_mode,
        semantic_weight=semantic_weight,
        cf_scores=cf_scores,
        attention_weight=attention_weight,
        cf_weight=cf_weight,
    )
    return {
        "sample_id": xfg_path,
        "label": label,
        "pred_prob": float(evidence["prob"]),
        "pred_label": pred_label,
        "true_vul_lines": true_lines,
        "label_source": label_source,
        "ranked_lines": [item["line"] for item in ranked],
        "top_lines": ranked[:5],
    }


def metrics_path_for(output_path: Path, limit: int) -> Path:
    return output_path.with_name(f"minimal_metrics_{limit}.csv")


def report_path_for(output_path: Path, limit: int) -> Path:
    return output_path.with_name(f"label_extraction_report_{limit}.json")


def write_metrics_csv(metrics_path: Path, metrics: Dict[str, float]):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def main():
    args = parse_args()

    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_split_paths(config, args.split, args.limit, args.data_json)
    vocab = load_vocabulary(config, args.w2v, args.vocab)
    model = build_model(config, vocab, args.checkpoint, device, args.strict_checkpoint)
    semantic_weight = resolve_semantic_weight(args.score_mode, args.semantic_weight)

    records = []
    report = LabelExtractionReport()
    for index, xfg_path in enumerate(paths):
        record = explain_one(
            model,
            config,
            vocab,
            xfg_path,
            device,
            args.prefer_explicit_labels,
            args.label_strategy,
            args.score_mode,
            semantic_weight,
            args.cf_top_k,
            args.attention_weight,
            args.cf_weight,
        )
        records.append(record)
        report.update(
            sample_id=record["sample_id"],
            label=record["label"],
            true_vul_lines=record["true_vul_lines"],
            reason=record["label_source"],
        )
        print(f"[{index + 1}/{len(paths)}] {record['sample_id']} prob={record['pred_prob']:.4f}")

    metrics = calculate_localization_metrics(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_metrics_csv(metrics_path_for(output_path, args.limit), metrics)
    report_dict = report.to_dict()
    if report_dict["negative_with_line_labels_should_be_zero"] > 0:
        print(
            "[warning] negative_with_line_labels_should_be_zero="
            f"{report_dict['negative_with_line_labels_should_be_zero']}"
        )
    report_path_for(output_path, args.limit).write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
