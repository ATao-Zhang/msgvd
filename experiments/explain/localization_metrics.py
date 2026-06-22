from typing import Dict, Iterable, List, Optional


def _first_hit_rank(ranked_lines: Iterable[int], true_lines: Iterable[int]) -> Optional[int]:
    true_set = {int(line) for line in true_lines}
    if not true_set:
        return None
    for rank, line in enumerate(ranked_lines, start=1):
        if int(line) in true_set:
            return rank
    return None


def calculate_localization_metrics(records: List[Dict]) -> Dict[str, float]:
    """Calculate Top-k, MRR, and IFA for vulnerable samples with line labels."""
    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    reciprocal_rank_sum = 0.0
    ifa_sum = 0.0

    for record in records:
        true_lines = record.get("true_vul_lines") or []
        if int(record.get("label", 0)) != 1 or not true_lines:
            continue
        ranked_lines = record.get("ranked_lines")
        if ranked_lines is None:
            ranked_lines = [item["line"] for item in record.get("top_lines", [])]
        if not ranked_lines:
            continue

        total += 1
        rank = _first_hit_rank(ranked_lines, true_lines)
        if rank is None:
            ifa = len(ranked_lines)
        else:
            top1 += int(rank <= 1)
            top3 += int(rank <= 3)
            top5 += int(rank <= 5)
            reciprocal_rank_sum += 1.0 / rank
            ifa = rank - 1
        ifa_sum += float(ifa)

    denom = max(total, 1)
    return {
        "evaluated_samples": total,
        "top1_acc": top1 / denom,
        "top3_acc": top3 / denom,
        "top5_acc": top5 / denom,
        "mrr": reciprocal_rank_sum / denom,
        "ifa": ifa_sum / denom,
    }
