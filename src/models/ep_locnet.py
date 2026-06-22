from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.models.vd import DeepWuKong


@dataclass
class EPLocNetOutput:
    detection_logits: torch.Tensor
    line_scores: torch.Tensor
    line_ids: torch.Tensor
    line_batch: torch.Tensor
    candidate_mask: torch.Tensor
    node_scores: torch.Tensor
    path_scores: torch.Tensor


class EPLocNet(nn.Module):
    """Execution Path-based Localization Network.

    EP-LocNet reuses the MSAVD backbone for function/path-level detection and
    adds a lightweight statement localization head.  It aligns node evidence
    from execution-path encoders to source lines, injects path risk as context,
    and produces ranked Top-k vulnerable statements.
    """

    def __init__(self,
                 backbone: DeepWuKong,
                 stmt_feature_size: int = 6,
                 hidden_size: Optional[int] = None,
                 candidate_top_ratio: float = 0.4,
                 consistency_weight: float = 0.1,
                 ranking_weight: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.node_dim = backbone.graph_hidden_size
        self.hidden_size = hidden_size or self.node_dim
        self.candidate_top_ratio = candidate_top_ratio
        self.consistency_weight = consistency_weight
        self.ranking_weight = ranking_weight
        self.context_gate = nn.Sequential(
            nn.Linear(self.node_dim + stmt_feature_size + 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.loc_head = nn.Linear(self.hidden_size, 1)

    @staticmethod
    def _line_keys(line_ids: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # PyG batches concatenate line ids from multiple functions; combine graph
        # id and source line id so identical line numbers in different functions
        # remain separate statements.
        offset = line_ids.max().clamp(min=0) + 1
        return batch * offset + line_ids

    @staticmethod
    def _aggregate_lines(values: torch.Tensor,
                         line_ids: torch.Tensor,
                         batch: torch.Tensor,
                         reduce: str = "mean") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = EPLocNet._line_keys(line_ids, batch)
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        if values.dim() == 1:
            out = values.new_zeros((unique_keys.numel(),))
        else:
            out = values.new_zeros((unique_keys.numel(), values.size(-1)))
        for idx in range(unique_keys.numel()):
            mask = inverse == idx
            selected = values[mask]
            if reduce == "max":
                out[idx] = selected.max(dim=0).values
            else:
                out[idx] = selected.mean(dim=0)
        line_out = line_ids.new_zeros(unique_keys.numel())
        batch_out = batch.new_zeros(unique_keys.numel())
        for idx in range(unique_keys.numel()):
            first = torch.nonzero(inverse == idx, as_tuple=False)[0, 0]
            line_out[idx] = line_ids[first]
            batch_out[idx] = batch[first]
        return out, line_out, batch_out

    def _candidate_mask(self,
                        line_node_scores: torch.Tensor,
                        line_stmt_features: torch.Tensor,
                        line_batch: torch.Tensor) -> torch.Tensor:
        dangerous_or_structural = line_stmt_features[:, [0, 1, 2, 4]].max(dim=1).values > 0
        candidate_mask = dangerous_or_structural.clone()
        for graph_id in torch.unique(line_batch):
            graph_mask = line_batch == graph_id
            graph_scores = line_node_scores[graph_mask]
            if graph_scores.numel() == 0:
                continue
            k = max(1, int(graph_scores.numel() * self.candidate_top_ratio))
            top_local = torch.topk(graph_scores, k=k).indices
            graph_indices = torch.nonzero(graph_mask, as_tuple=False).flatten()
            candidate_mask[graph_indices[top_local]] = True
        return candidate_mask

    def forward(self, batch: Batch) -> EPLocNetOutput:
        evidence = self.backbone.forward_with_evidence(batch)
        node_embeddings = evidence["node_embeddings"]
        raw_node_scores = evidence["node_scores"]
        node_scores = torch.sigmoid(raw_node_scores)
        node_batch = evidence["node_batch"]
        line_ids = batch.line_ids
        stmt_features = batch.stmt_features

        line_node_repr, line_out, line_batch = self._aggregate_lines(
            node_embeddings, line_ids, node_batch, reduce="mean")
        line_node_scores, _, _ = self._aggregate_lines(
            node_scores, line_ids, node_batch, reduce="max")
        line_stmt_features, _, _ = self._aggregate_lines(
            stmt_features, line_ids, node_batch, reduce="max")

        path_scores = evidence["path_scores"]
        path_context = path_scores[line_batch].unsqueeze(-1)
        occurrence_context = line_node_scores.unsqueeze(-1)
        candidate_mask = self._candidate_mask(line_node_scores, line_stmt_features, line_batch)
        fused = torch.cat([
            line_node_repr,
            line_stmt_features,
            path_context,
            occurrence_context,
        ], dim=-1)
        hidden = self.context_gate(fused)
        line_logits = self.loc_head(hidden).squeeze(-1)
        # Non-candidates stay rankable but receive a prior penalty so the search
        # space is focused before final scoring, mirroring slice/subgraph pruning.
        line_logits = line_logits - (~candidate_mask).float() * 2.0
        line_scores = torch.sigmoid(line_logits)
        return EPLocNetOutput(
            detection_logits=evidence["logits"],
            line_scores=line_scores,
            line_ids=line_out,
            line_batch=line_batch,
            candidate_mask=candidate_mask,
            node_scores=node_scores,
            path_scores=path_scores,
        )

    def localization_targets(self, batch: Batch, output: EPLocNetOutput) -> torch.Tensor:
        node_targets = batch.node_vul_labels.float()
        line_targets, _, _ = self._aggregate_lines(
            node_targets, batch.line_ids, batch.batch, reduce="max")
        return line_targets.to(output.line_scores.device)

    def loss(self,
             batch: Batch,
             labels: torch.Tensor,
             output: Optional[EPLocNetOutput] = None) -> Dict[str, torch.Tensor]:
        output = output or self.forward(batch)
        det_loss = F.cross_entropy(output.detection_logits, labels)
        line_targets = self.localization_targets(batch, output)
        if line_targets.sum() > 0:
            loc_loss = F.binary_cross_entropy(output.line_scores, line_targets)
        else:
            loc_loss = output.line_scores.mean() * 0.0
        graph_max_scores = []
        for graph_id in torch.unique(output.line_batch):
            graph_line_scores = output.line_scores[output.line_batch == graph_id]
            graph_max_scores.append(graph_line_scores.max())
        graph_max_scores = torch.stack(graph_max_scores) if graph_max_scores else output.path_scores.new_zeros(0)
        det_probs = torch.softmax(output.detection_logits, dim=-1)[:, 1]
        consistency_loss = F.mse_loss(graph_max_scores, det_probs[:graph_max_scores.numel()])
        rank_loss = self._ranking_loss(output.line_scores, line_targets, output.line_batch)
        total = det_loss + loc_loss + self.consistency_weight * consistency_loss + self.ranking_weight * rank_loss
        return {
            "loss": total,
            "det_loss": det_loss,
            "loc_loss": loc_loss,
            "consistency_loss": consistency_loss,
            "rank_loss": rank_loss,
        }

    @staticmethod
    def _ranking_loss(line_scores: torch.Tensor,
                      line_targets: torch.Tensor,
                      line_batch: torch.Tensor,
                      margin: float = 0.2) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for graph_id in torch.unique(line_batch):
            mask = line_batch == graph_id
            pos = line_scores[mask & (line_targets > 0)]
            neg = line_scores[mask & (line_targets <= 0)]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            losses.append(F.relu(margin - pos.max() + neg.max()))
        if not losses:
            return line_scores.mean() * 0.0
        return torch.stack(losses).mean()

    def topk(self, output: EPLocNetOutput, k: int = 5) -> List[List[Tuple[int, float]]]:
        results: List[List[Tuple[int, float]]] = []
        for graph_id in torch.unique(output.line_batch):
            mask = output.line_batch == graph_id
            scores = output.line_scores[mask]
            lines = output.line_ids[mask]
            top = torch.topk(scores, k=min(k, scores.numel()))
            results.append([(int(lines[idx].item()), float(scores[idx].item())) for idx in top.indices])
        return results
