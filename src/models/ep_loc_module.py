from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.optim import Adam, Adamax, RMSprop, SGD

from src.datas.samples import XFGBatch
from src.metrics import LocalizationStatistic, Statistic
from src.models.ep_locnet import EPLocNet, EPLocNetOutput
from src.models.vd import DeepWuKong
from src.vocabulary import Vocabulary


class EPLocModule(LightningModule):
    """Lightning wrapper for training and evaluating EP-LocNet.

    The module loads a trained MSAVD/DeepWuKong checkpoint as the detection
    backbone, then trains the EP-LocNet statement localization head.  By
    default the backbone is frozen for the second-stage training described in
    the thesis workflow; pass ``freeze_backbone=False`` for joint fine-tuning.
    """

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax,
    }

    def __init__(self,
                 config: DictConfig,
                 vocab: Vocabulary,
                 backbone_checkpoint: str,
                 backbone: Optional[DeepWuKong] = None,
                 freeze_backbone: bool = True,
                 top_k: Optional[int] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.__config = config
        self.__vocab = vocab
        self.__backbone_checkpoint = backbone_checkpoint
        self.__freeze_backbone = freeze_backbone
        self.__top_k = top_k or config.localization.top_k
        if backbone is None:
            backbone = DeepWuKong.load_from_checkpoint(backbone_checkpoint)
        self.__loc_model = EPLocNet(
            backbone=backbone,
            stmt_feature_size=config.localization.stmt_feature_size,
            hidden_size=config.localization.hidden_size,
            candidate_top_ratio=config.localization.candidate_top_ratio,
            consistency_weight=config.localization.consistency_weight,
            ranking_weight=config.localization.ranking_weight,
        )
        if freeze_backbone:
            self.freeze_backbone()

    @property
    def config(self) -> DictConfig:
        return self.__config

    @property
    def vocab(self) -> Vocabulary:
        return self.__vocab

    @property
    def backbone_checkpoint(self) -> str:
        return self.__backbone_checkpoint

    @property
    def top_k(self) -> int:
        return self.__top_k

    @top_k.setter
    def top_k(self, value: int):
        self.__top_k = value

    def freeze_backbone(self):
        for parameter in self.__loc_model.backbone.parameters():
            parameter.requires_grad = False
        self.__loc_model.backbone.eval()

    def unfreeze_backbone(self):
        for parameter in self.__loc_model.backbone.parameters():
            parameter.requires_grad = True
        self.__freeze_backbone = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.__freeze_backbone:
            self.__loc_model.backbone.eval()
        return self

    def forward(self, batch) -> EPLocNetOutput:  # type: ignore
        return self.__loc_model(batch)

    def topk(self, output: EPLocNetOutput, k: Optional[int] = None):
        return self.__loc_model.topk(output, k or self.__top_k)

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:
        optimizer_cls = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = optimizer_cls(parameters,
                                  self.__config.hyper_parameters.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(self, batch: XFGBatch, group: str) -> Dict:
        output = self(batch.graphs)
        losses = self.__loc_model.loss(batch.graphs, batch.labels, output)
        with torch.no_grad():
            _, preds = output.detection_logits.max(dim=1)
            detection_stat = Statistic.calculate_statistic(batch.labels,
                                                           preds,
                                                           2)
            loc_stat = self._localization_statistic(batch.graphs, output)
            logs = {f"{group}_{key}": value for key, value in losses.items()}
            logs.update(detection_stat.calculate_metrics(f"{group}_det"))
            logs.update(loc_stat.calculate_metrics(f"{group}_loc"))
        return {
            "loss": losses["loss"],
            "logs": logs,
            "detection_stat": detection_stat,
            "loc_stat": loc_stat,
        }

    def _localization_statistic(self, graphs,
                                output: EPLocNetOutput) -> LocalizationStatistic:
        line_targets = self.__loc_model.localization_targets(graphs, output)
        statistic = LocalizationStatistic()
        for graph_id in torch.unique(output.line_batch):
            mask = output.line_batch == graph_id
            scores = output.line_scores[mask]
            lines = output.line_ids[mask]
            targets = line_targets[mask]
            if targets.sum() <= 0:
                continue
            ordered = torch.argsort(scores, descending=True)
            ranked_lines = [int(lines[idx].item()) for idx in ordered]
            target_lines = [
                int(line.item())
                for line in lines[targets > 0]
            ]
            statistic.update_from_ranked_lines(ranked_lines, target_lines)
        return statistic

    def training_step(self, batch: XFGBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        result = self._shared_step(batch, "train")
        self.log_dict(result["logs"], on_step=True, on_epoch=False)
        self.log("F1",
                 result["logs"].get("train_det_f1", 0.0),
                 prog_bar=True,
                 logger=False)
        return {"loss": result["loss"],
                "detection_stat": result["detection_stat"],
                "loc_stat": result["loc_stat"]}

    def validation_step(self, batch: XFGBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        return self._shared_step(batch, "val")

    def test_step(self, batch: XFGBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        return self._shared_step(batch, "test")

    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        detection_stat = Statistic.union_statistics(
            [out["detection_stat"] for out in step_outputs])
        loc_stat = LocalizationStatistic.union_statistics(
            [out["loc_stat"] for out in step_outputs])
        log.update(detection_stat.calculate_metrics(f"{group}_det"))
        log.update(loc_stat.calculate_metrics(f"{group}_loc"))
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
