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


def _mask_tensor_rows(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = tensor.clone()
    row_mask = mask.to(masked.device)
    masked[row_mask] = 0
    return masked


def _fallback_non_pad_token(row: torch.Tensor, pad_idx: int, global_fallback: int = None):
    non_pad = row[row != int(pad_idx)]
    if non_pad.numel() > 0:
        return int(non_pad[0].item())
    if global_fallback is not None and int(global_fallback) != int(pad_idx):
        return int(global_fallback)
    return None


def mask_node_tokens_keep_length(
    x: torch.Tensor,
    node_mask: torch.Tensor,
    pad_idx: int,
    mask_idx: int = None,
) -> Tuple[torch.Tensor, bool, str]:
    """Mask selected node token sequences without creating zero-length samples.

    For each selected node, original PAD positions stay PAD. Original non-PAD
    positions are replaced with UNK/MASK when available. If no mask token is
    available, the row's first original non-PAD token is used as a legal
    non-PAD replacement. This keeps the RNN/ST encoder sequence length > 0.
    """
    if not isinstance(x, torch.Tensor):
        return x, False, "Data.x is not a tensor; cannot mask node token features"
    if x.dim() != 2:
        return x, False, f"Data.x dim is {x.dim()}, expected 2D [num_nodes, seq_len]; skip counterfactual"
    if x.size(1) == 0:
        return x, False, "Data.x has seq_len=0; cannot build length-preserving mask"

    masked = x.clone()
    row_mask = node_mask.to(masked.device)
    selected_rows = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
    if not selected_rows:
        return masked, False, "node mask selected no rows"

    pad_idx = int(pad_idx)
    if mask_idx is not None and int(mask_idx) == pad_idx:
        mask_idx = None

    all_non_pad = masked[masked != pad_idx]
    global_fallback = int(all_non_pad[0].item()) if all_non_pad.numel() > 0 else None

    for row_index in selected_rows:
        row = masked[row_index]
        non_pad_positions = row != pad_idx
        replacement = mask_idx
        if replacement is None:
            replacement = _fallback_non_pad_token(row, pad_idx, global_fallback)
        if replacement is None:
            return x, False, "cannot find a legal non-PAD replacement token for counterfactual mask"

        if int(non_pad_positions.sum().item()) == 0:
            row[0] = int(replacement)
        else:
            row[non_pad_positions] = int(replacement)
        masked[row_index] = row

    return masked, True, "masked non-PAD token ids with length-preserving UNK/MASK fallback"


def mask_line_features(data, line_no: int, pad_id: int = None, mask_id: int = None) -> Tuple[object, bool, str]:
    """Return a cloned graph with token-level features for one source line masked.

    The MSAVD/XFG graph stores token ids in ``Data.x`` and source line ids in
    ``Data.line_ids``. For a target source line, this masks all matching nodes
    with a length-preserving token perturbation: PAD stays PAD, non-PAD tokens
    become UNK/MASK or another legal non-PAD fallback. Optional statement
    features for the same nodes are zeroed.
    """
    if not hasattr(data, "line_ids"):
        return data, False, "data has no line_ids; cannot locate nodes for source line"
    if not hasattr(data, "x"):
        return data, False, "data has no x tensor; cannot mask node token features"
    if pad_id is None:
        return data, False, "pad_id is required for length-preserving token mask"

    node_mask = _line_node_mask(data, line_no)
    if node_mask is None or int(node_mask.sum().item()) == 0:
        return data, False, f"line {line_no} has no matching graph nodes"

    masked_data = _clone_data(data)
    masked_x, ok, reason = mask_node_tokens_keep_length(masked_data.x, node_mask, pad_id, mask_id)
    if not ok:
        return data, False, reason
    masked_data.x = masked_x
    if hasattr(masked_data, "stmt_features") and isinstance(masked_data.stmt_features, torch.Tensor):
        masked_data.stmt_features = _mask_tensor_rows(masked_data.stmt_features, node_mask)
    return masked_data, True, reason + "; stmt_features zeroed when present"


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
    mask_id: int = None,
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
        masked_data, ok, reason = mask_line_features(data, line_no, pad_id=pad_id, mask_id=mask_id)
        if not ok:
            results[line_no] = dict(CF_EMPTY_SCORE)
            warnings.append(reason)
            continue
        try:
            masked_prob = vulnerable_probability(model, masked_data, device)
        except Exception as exc:  # noqa: BLE001 - keep full experiment running on one-line CF failure.
            results[line_no] = dict(CF_EMPTY_SCORE)
            warnings.append(f"cf failed for line {line_no}: {exc}")
            continue
        prob_drop = max(0.0, float(original_prob) - float(masked_prob))
        results[line_no] = {
            "cf_score": prob_drop,
            "masked_prob": masked_prob,
            "prob_drop": prob_drop,
        }
    return results, warnings
