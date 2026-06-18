from typing import Dict, Iterable, List, Tuple

import torch
from torch_geometric.data import Batch


CF_EMPTY_SCORE = {
    "cf_score": 0.0,
    "masked_prob": None,
    "prob_drop": 0.0,
}


def _clone_data(data):
    if hasattr(data, "clone"):
        return data.clone()
    raise TypeError("PyG Data object does not support clone(); cannot run counterfactual mask.")


def _line_node_mask(data, line_no: int):
    if not hasattr(data, "line_ids"):
        return None
    return data.line_ids.detach().cpu() == int(line_no)


def _mask_tensor_rows(tensor: torch.Tensor, mask: torch.Tensor, pad_id: int = None) -> torch.Tensor:
    masked = tensor.clone()
    row_mask = mask.to(masked.device)
    if masked.dtype in (torch.long, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        fill_value = 0 if pad_id is None else int(pad_id)
        masked[row_mask] = fill_value
    else:
        masked[row_mask] = 0
    return masked


def mask_line_features(data, line_no: int, pad_id: int = None) -> Tuple[object, bool, str]:
    """Return a cloned graph with features for one source line masked.

    The MSAVD/XFG graph stores token ids in ``Data.x`` and source line ids in
    ``Data.line_ids``. For a target source line, this masks all matching nodes by
    replacing token ids with PAD and clearing optional statement features.
    """
    if not hasattr(data, "line_ids"):
        return data, False, "data has no line_ids; cannot locate nodes for source line"
    if not hasattr(data, "x"):
        return data, False, "data has no x tensor; cannot mask node token features"

    node_mask = _line_node_mask(data, line_no)
    if node_mask is None or int(node_mask.sum().item()) == 0:
        return data, False, f"line {line_no} has no matching graph nodes"

    masked_data = _clone_data(data)
    masked_data.x = _mask_tensor_rows(masked_data.x, node_mask, pad_id=pad_id)
    if hasattr(masked_data, "stmt_features") and isinstance(masked_data.stmt_features, torch.Tensor):
        masked_data.stmt_features = _mask_tensor_rows(masked_data.stmt_features, node_mask, pad_id=None)
    return masked_data, True, "masked Data.x token ids with PAD and stmt_features with zeros"


def vulnerable_probability(model, data, device) -> float:
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        logits = model(batch)
        prob = torch.softmax(logits, dim=-1)[0, 1]
    return float(prob.detach().cpu())


def compute_counterfactual_scores(
    model,
    data,
    candidate_lines: Iterable[int],
    original_prob: float,
    device,
    pad_id: int = None,
    cf_top_k: int = 10,
) -> Tuple[Dict[int, Dict], List[str]]:
    """Compute lightweight line-level counterfactual scores for candidate lines.

    cf_score(line) = max(0, P_vul(original) - P_vul(mask_line)). Only the first
    ``cf_top_k`` unique candidate lines are evaluated to keep the experiment
    runnable on the full split.
    """
    results: Dict[int, Dict] = {}
    warnings: List[str] = []
    seen = set()
    selected: List[int] = []
    for line in candidate_lines:
        line_no = int(line)
        if line_no in seen:
            continue
        seen.add(line_no)
        selected.append(line_no)
        if len(selected) >= int(cf_top_k):
            break

    for line_no in selected:
        masked_data, ok, reason = mask_line_features(data, line_no, pad_id=pad_id)
        if not ok:
            results[line_no] = dict(CF_EMPTY_SCORE)
            warnings.append(reason)
            continue
        masked_prob = vulnerable_probability(model, masked_data, device)
        prob_drop = max(0.0, float(original_prob) - float(masked_prob))
        results[line_no] = {
            "cf_score": prob_drop,
            "masked_prob": masked_prob,
            "prob_drop": prob_drop,
        }
    return results, warnings
