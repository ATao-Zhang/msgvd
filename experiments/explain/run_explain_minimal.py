import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

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
from src.datas.graphs import XFG  # noqa: E402
from src.models.vd import DeepWuKong  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MSAVD execution-path evidence localization.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained MSAVD checkpoint.")
    parser.add_argument("--config", default="configs/dwk.yaml", help="Path to the YAML config.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of samples to run.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--vocab", default=None, help="Optional pickled vocabulary path.")
    parser.add_argument("--prefer-explicit-labels", action="store_true",
                        help="Prefer graph metadata line labels before weak SARD/Juliet extraction.")
    return parser.parse_args()


def load_split_paths(config, split: str, limit: int) -> List[str]:
    split_path = ROOT / config.data_folder / config.dataset.name / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    paths = json.loads(split_path.read_text(encoding="utf-8"))
    if limit is not None and limit > 0:
        paths = paths[:limit]
    return [str(path) for path in paths]


def load_vocabulary(config, vocab_path: str = None) -> Vocabulary:
    if vocab_path:
        return Vocabulary.load_vocabulary(vocab_path)
    return Vocabulary.build_from_w2v(config.gnn.w2v_path)


def build_model(config, vocab: Vocabulary, checkpoint_path: str, device: torch.device) -> DeepWuKong:
    model = DeepWuKong(config, vocab, vocab.get_vocab_size(), vocab.get_pad_id())
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned_state = {}
    for key, value in state_dict.items():
        cleaned_key = key[6:] if key.startswith("model.") else key
        cleaned_state[cleaned_key] = value
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[warn] missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected checkpoint keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().float().cpu()
    if scores.numel() == 0:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if torch.isclose(max_score, min_score):
        return torch.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _evidence_scores(model: DeepWuKong, graph_batch: Batch) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
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


def line_rankings(node_scores: torch.Tensor, line_ids: torch.Tensor, code_by_line: Dict[int, str]) -> List[Dict]:
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


def explain_one(model: DeepWuKong, config, vocab: Vocabulary, xfg_path: str, device: torch.device,
                prefer_explicit_labels: bool) -> Dict:
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
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_split_paths(config, args.split, args.limit)
    vocab = load_vocabulary(config, args.vocab)
    model = build_model(config, vocab, args.checkpoint, device)

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
