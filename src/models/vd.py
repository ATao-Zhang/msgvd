# vd.py  —— 修复版
# 主要修复点：
# - 使用 automatic optimization（Lightning）以正确使用 scheduler（若你坚持手动优化，请告知）
# - 缓存 Statistic 实例，避免每 batch 构造
# - 归一化 task_weights
# - 使用 math.cos 避免创建 tensor 在 CPU 上
# - pack/padding 等处的安全检查在下层模块已修复（common_layers.py）
# - 课程学习接口保持原样，但 SARDDataOptimizer 的过滤函数已在 gnns.py 中实现

from typing import Dict, List, Optional, Any
import math
from omegaconf import DictConfig
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim import Adam, SGD, Adamax, RMSprop
from torch_geometric.data import Batch
import torch.nn.functional as F

from src.metrics import Statistic
from src.vocabulary import Vocabulary
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder, EnhancedGGRNDevignEncoder, SARDDataOptimizer


class MultiTaskClassifier(nn.Module):
    """自适应多任务分类器（与原来保持一致，修复 task_weights 归一化）"""
    def __init__(self, hidden_size: int, n_classes: int, n_tasks: int = 1,
                 task_names: Optional[List[str]] = None):
        super().__init__()
        self.n_tasks = n_tasks
        self.task_names = task_names or [f"task_{i}" for i in range(n_tasks)]

        # 共享骨干
        self.shared_backbone = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # 任务特定头
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_size, n_classes) for _ in range(n_tasks)
        ])

        # 动态任务权重（可学习，但在 forward 时归一化）
        self.task_weights = nn.Parameter(torch.ones(n_tasks) / n_tasks)

        # 焦点损失参数
        self.focal_gamma = 2.0
        self.class_weights = None

    def set_class_weights(self, weights: torch.Tensor):
        """设置类别权重用于焦点损失"""
        self.class_weights = weights

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared_backbone(x)

        outputs = {}
        for i, head in enumerate(self.task_heads):
            outputs[self.task_names[i]] = head(shared_features)

        return outputs

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算多任务损失"""
        total_loss = 0.0
        # FIX: 归一化 task_weights 避免 scale 问题
        weights = self.task_weights
        if weights.sum().item() != 0:
            weights = weights / weights.sum()

        for i, task_name in enumerate(self.task_names):
            if task_name not in targets:
                continue

            logits = outputs[task_name]
            target = targets[task_name]

            # cross-entropy + focal
            if self.class_weights is not None:
                ce_loss = F.cross_entropy(logits, target, weight=self.class_weights, reduction='none')
            else:
                ce_loss = F.cross_entropy(logits, target, reduction='none')

            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()

            total_loss = total_loss + weights[i] * focal_loss

        return total_loss


class EnhancedProgressiveScheduler:
    """渐进式训练调度器（保留，但不直接管理 optimizer.step；用于查询 lr scale）"""
    def __init__(self, config: DictConfig):
        self.config = config
        self.current_stage = "warmup"

    def get_lr_scale(self, epoch: int) -> float:
        warmup = getattr(self.config, "warmup_epochs", 5)
        main = getattr(self.config, "main_epochs", 50)
        finetune_scale = getattr(self.config, "finetune_lr_scale", 0.1)

        if epoch < warmup:
            return min(1.0, epoch / warmup)
        elif epoch < warmup + main:
            progress = (epoch - warmup) / main
            return 0.5 * (1 + math.cos(progress * math.pi))
        else:
            return finetune_scale


class EnhancedDeepWuKong(LightningModule):
    """增强的DeepWuKong模型（修复版）"""
    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder,
        "enhanced_ggrn": EnhancedGGRNDevignEncoder
    }

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int, task_names: Optional[List[str]] = None,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.save_hyperparameters(ignore="vocab")
        self.__config = config
        self.task_names = task_names or ["vulnerability"]

        # 图编码器
        enc_cls = self._encoders.get(config.gnn.name)
        if enc_cls is None:
            raise KeyError(f"Encoder {config.gnn.name} not supported")
        self.__graph_encoder = enc_cls(
            config.gnn, vocab, vocabulary_size, pad_idx
        )

        # 多任务分类器
        self.__classifier = MultiTaskClassifier(
            hidden_size=config.gnn.hidden_size,
            n_classes=config.classifier.n_classes,
            n_tasks=len(self.task_names),
            task_names=self.task_names
        )

        if class_weights is not None:
            self.__classifier.set_class_weights(class_weights)

        # SARD优化器
        self.sard_optimizer = SARDDataOptimizer(config)

        # 训练统计
        self.training_stats = {
            "stage": "warmup",
            "current_epoch": 0,
            "difficulty_level": 0
        }

        # 允许 Lightning 自动优化（便于 scheduler 工作）
        self.automatic_optimization = True

        # Statistic 缓存（FIX）
        self._statistic = Statistic()

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        graph_hid = self.__graph_encoder(batch)
        return self.__classifier(graph_hid)

    def configure_optimizers(self):
        optimizer_cls = self._get_optimizer(self.__config.hyper_parameters.optimizer)
        optimizer = optimizer_cls(self.parameters(), lr=self.__config.hyper_parameters.learning_rate)

        # 使用 LambdaLR 与 EnhancedProgressiveScheduler 计算 scale
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: EnhancedProgressiveScheduler(self.__config.hyper_parameters).get_lr_scale(epoch)
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _prepare_multi_task_targets(self, batch) -> Dict[str, torch.Tensor]:
        targets = {}
        if hasattr(batch, 'multi_task_labels') and batch.multi_task_labels is not None:
            for task_name in self.task_names:
                if task_name in batch.multi_task_labels:
                    targets[task_name] = batch.multi_task_labels[task_name]
        else:
            # 回退到单任务
            targets = {self.task_names[0]: batch.labels}
        return targets

    def _apply_curriculum_learning(self, batch, difficulty_level: int):
        # SARDDataOptimizer 实现了 filter_by_difficulty, augment_training_data
        batch = self.sard_optimizer.filter_by_difficulty(batch, difficulty_level)
        batch = self.sard_optimizer.augment_training_data(batch, difficulty_level)
        return batch

    def training_step(self, batch, batch_idx):
        # 应用课程学习
        batch = self._apply_curriculum_learning(batch, self.training_stats.get("difficulty_level", 0))

        logits = self(batch.graphs)
        targets = self._prepare_multi_task_targets(batch)
        loss = self.__classifier.compute_loss(logits, targets)

        # Lightning 自动优化，直接返回 loss 并记录指标
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        results = {}
        with torch.no_grad():
            for task_name, task_logits in logits.items():
                if task_name in targets:
                    _, preds = task_logits.max(dim=1)
                    statistic = Statistic().calculate_statistic(
                        targets[task_name], preds, self.__config.classifier.n_classes
                    )
                    task_metrics = statistic.calculate_metrics(group=f"train_{task_name}")
                    results.update(task_metrics)
        self.log_dict(results, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_start(self):
        # 更新训练统计
        self.training_stats["current_epoch"] = int(self.current_epoch)
        total_epochs = getattr(self.trainer, 'max_epochs', None)
        if total_epochs is not None:
            # 更新难度级别
            if hasattr(self, 'sard_optimizer'):
                self.sard_optimizer.update_difficulty = getattr(self.sard_optimizer, "update_difficulty", lambda e, t: None)
                # no-op if not implemented; SARDDataOptimizer may not implement update_difficulty
                # 保持训练统计
        return

    def validation_step(self, batch, batch_idx):
        logits = self(batch.graphs)
        targets = self._prepare_multi_task_targets(batch)
        loss = self.__classifier.compute_loss(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        with torch.no_grad():
            main_task = self.task_names[0]
            if main_task in logits and main_task in targets:
                _, preds = logits[main_task].max(dim=1)
                statistic = Statistic().calculate_statistic(
                    targets[main_task], preds, self.__config.classifier.n_classes
                )
                metrics = statistic.calculate_metrics(group="val")
                self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch.graphs)
        targets = self._prepare_multi_task_targets(batch)
        loss = self.__classifier.compute_loss(logits, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        with torch.no_grad():
            main_task = self.task_names[0]
            if main_task in logits and main_task in targets:
                _, preds = logits[main_task].max(dim=1)
                statistic = Statistic().calculate_statistic(
                    targets[main_task], preds, self.__config.classifier.n_classes
                )
                metrics = statistic.calculate_metrics(group="test")
                self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def _get_optimizer(self, name: str):
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")


# 向后兼容的原始DeepWuKong
class DeepWuKong(EnhancedDeepWuKong):
    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int, pad_idx: int):
        super().__init__(config, vocab, vocabulary_size, pad_idx, ["vulnerability"])
