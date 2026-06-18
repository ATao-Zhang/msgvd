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

from experiments.explain.label_extractor import (  # noqa: E402
    LabelExtractionReport,
    extract_vulnerable_lines,
    graph_file_path,
    read_source_lines,
    source_line_map,
)
from experiments.explain.localization_metrics import calculate_localization_metrics  # noqa: E402


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
    parser.add_argument("--prefer-explicit-labels", action="store_true",
                        help="Prefer graph metadata line labels before weak SARD/Juliet extraction.")
    return parser.parse_args()


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


def line_rankings(node_scores, line_ids, code_by_line: Dict[int, str]) -> List[Dict]:
    line_scores: Dict[int, float] = {}
    for score, line_no in zip(node_scores.tolist(), line_ids.detach().cpu().tolist()):
        line_no = int(line_no)
        line_scores[line_no] = max(line_scores.get(line_no, float("-inf")), float(score))
    ranked = sorted(line_scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "line": int(line_no),
            "score": float(score),
            "code": code_by_line.get(int(line_no), ""),
        }
        for line_no, score in ranked
    ]


def explain_one(model, config, vocab, xfg_path: str, device, prefer_explicit_labels: bool) -> Dict:
    import networkx as nx
    from torch_geometric.data import Batch

    from src.datas.graphs import XFG

    graph_nx = nx.read_gpickle(xfg_path)
    label = int(graph_nx.graph["label"])
    source_lines = read_source_lines(graph_file_path(graph_nx))
    code_by_line = source_line_map(graph_nx, source_lines)
    true_lines, label_source = extract_vulnerable_lines(
        graph_nx,
        source_lines=source_lines,
        prefer_explicit=prefer_explicit_labels,
    )

    xfg = XFG(xfg=graph_nx)
    data = xfg.to_torch(vocab, config.dataset.token.max_parts)
    graph_batch = Batch.from_data_list([data]).to(device)

    with torch.no_grad():
        node_scores, pred_label, evidence = _evidence_scores(model, graph_batch)

    ranked = line_rankings(node_scores, data.line_ids, code_by_line)
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

    records = []
    report = LabelExtractionReport()
    for index, xfg_path in enumerate(paths):
        record = explain_one(model, config, vocab, xfg_path, device, args.prefer_explicit_labels)
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
    report_path_for(output_path, args.limit).write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
