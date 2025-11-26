from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention
import torch.nn.functional as F
from src.vocabulary import Vocabulary
from src.models.modules.common_layers import EnhancedSTEncoder, STEncoder
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Dict, List
from torch_geometric.utils import to_dense_batch


class EnhancedMultiRelationalGGRN(nn.Module):
    """增强的多关系GGRN，支持边类型感知和门控跳跃连接"""
    def __init__(self, hidden_size: int, num_steps: int = 3, num_relations: int = 1):
        super().__init__()
        self.hidden = hidden_size
        self.T = num_steps
        self.num_rel = num_relations
        
        # 边类型特定的变换矩阵
        self.W = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False)
                                for _ in range(num_relations)])
        
        # 门控机制
        self.gates = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) 
                                   for _ in range(num_steps)])
        
        # 注意力融合
        self.layer_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.layer_weights = nn.Parameter(torch.ones(num_steps) / num_steps)
        
        # 残差连接
        self.residual_norm = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_type: Optional[torch.Tensor] = None):
        if edge_type is None:
            edge_type = h.new_zeros(edge_index.size(1), dtype=torch.long)
            
        R = int(edge_type.max().item()) + 1
        if R > self.num_rel:
            for _ in range(R - self.num_rel):
                self.W.append(nn.Linear(self.hidden, self.hidden, bias=False).to(h.device))
            self.num_rel = R

        layer_outputs = []
        hi = h
        
        for step in range(self.T):
            m = torch.zeros_like(hi)
            for r in range(self.num_rel):
                mask = (edge_type == r)
                if not torch.any(mask):
                    continue
                ei = edge_index[:, mask]
                src, dst = ei[0], ei[1]
                msg_src = self.W[r](hi)
                m.index_add_(0, dst, msg_src[src])
            
            # 门控更新
            hi = self.gates[step](m, hi)
            layer_outputs.append(hi.unsqueeze(1))

        # 多层特征注意力融合
        if len(layer_outputs) > 1:
            layer_stacked = torch.cat(layer_outputs, dim=1)  # [N, T, H]
            attn_output, _ = self.layer_attention(layer_stacked, layer_stacked, layer_stacked)
            
            # 加权融合
            weights = torch.softmax(self.layer_weights[:len(layer_outputs)], dim=0)
            weighted_output = sum(w * out.squeeze(1) for w, out in zip(weights, layer_outputs))
            
            # 残差连接
            final_output = self.residual_norm(weighted_output + h)
        else:
            final_output = hi

        return final_output


class HierarchicalAttentionPooling(nn.Module):
    """层次化注意力池化"""
    def __init__(self, hidden_size: int, num_perspectives: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_perspectives = num_perspectives
        
        # 节点级自注意力
        self.node_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # 多视角注意力
        self.perspective_attentions = nn.ModuleList([
            GlobalAttention(nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )) for _ in range(num_perspectives)
        ])
        
        # 视角融合
        self.perspective_fusion = nn.Linear(hidden_size * num_perspectives, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # 将节点特征转换为批次格式 [batch_size, seq_len, hidden_size]
        x_dense, mask = to_dense_batch(x, batch, fill_value=0)
        
        # 节点级自注意力
        attn_output, _ = self.node_attention(x_dense, x_dense, x_dense, key_padding_mask=~mask)
        enhanced_x_dense = x_dense + attn_output  # 残差连接
        
        # 将密集格式转换回节点格式
        batch_size, max_nodes, hidden_size = enhanced_x_dense.shape
        enhanced_x = enhanced_x_dense.reshape(-1, hidden_size)
        # 只保留实际存在的节点
        enhanced_x = enhanced_x[mask.reshape(-1)]
        
        # 多视角注意力
        perspective_outputs = []
        for attention in self.perspective_attentions:
            perspective_output = attention(enhanced_x, batch)
            perspective_outputs.append(perspective_output)
        
        # 融合多视角
        fused_perspectives = torch.cat(perspective_outputs, dim=1)
        graph_embedding = self.perspective_fusion(fused_perspectives)
        
        return self.layer_norm(graph_embedding)


class EnhancedDevignConvModule(nn.Module):
    """增强的Devign卷积模块"""
    def __init__(self, dim_h: int, dim_x0: int, proj_dim: int, c1: int = 64, c2: int = 128):
        super().__init__()
        
        # 增强的卷积路径A
        self.convA = nn.Sequential(
            nn.Conv1d(dim_h + dim_x0, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 增强的卷积路径B
        self.convB = nn.Sequential(
            nn.Conv1d(dim_h, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 增强的MLP
        self.mlpA = nn.Sequential(
            nn.Linear(c2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )
        
        self.mlpB = nn.Sequential(
            nn.Linear(c2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim)
        )

    @staticmethod
    def _pack_by_graph(x: torch.Tensor, batch: torch.Tensor):
        B = int(batch.max().item()) + 1
        sizes = torch.bincount(batch, minlength=B).tolist()
        splits = torch.split(x, sizes, dim=0)
        padded = pad_sequence(splits, batch_first=True)
        return padded.transpose(1, 2)

    def forward(self, H_T: torch.Tensor, X0: torch.Tensor, batch: torch.Tensor):
        seqH = self._pack_by_graph(H_T, batch)
        seqX = self._pack_by_graph(X0, batch)
        
        z = self.convA(torch.cat([seqH, seqX], dim=1)).squeeze(-1)
        y = self.convB(seqH).squeeze(-1)
        
        a, b = self.mlpA(z), self.mlpB(y)
        return a * b  # 门控机制


class EnhancedGGRNDevignEncoder(nn.Module):
    """增强的GGRN-Devign编码器"""
    def __init__(self, config, vocab, vocabulary_size, pad_idx):
        super().__init__()
        self.cfg = config
        
        # 使用增强的语句编码器
        self.st = EnhancedSTEncoder(config, vocab, vocabulary_size, pad_idx)
        
        self.proj_in = nn.Linear(config.rnn.hidden_size, config.hidden_size)
        
        # 增强的GGRN
        self.ggrn = EnhancedMultiRelationalGGRN(
            hidden_size=config.hidden_size,
            num_steps=getattr(config, "n_gru", 3),
            num_relations=getattr(config, "num_relations", 1),
        )
        
        # 时序融合层 - 双向GRU
        self.temporal_fusion = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        self.devign_conv = EnhancedDevignConvModule(
            dim_h=config.hidden_size, 
            dim_x0=config.hidden_size, 
            proj_dim=config.hidden_size
        )
        
        # 层次化注意力池化
        self.hierarchical_attention = HierarchicalAttentionPooling(config.hidden_size)
        
        self.fuse = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(getattr(config, "dropout", 0.2))
        )

    def forward(self, g: Batch) -> torch.Tensor:
        # 语句编码
        x0 = self.st(g.x)                 
        x0 = F.relu(self.proj_in(x0))
        
        # 图语义传播
        H_T = self.ggrn(x0, g.edge_index, getattr(g, "edge_type", None))
        
        # 时序融合
        temporal_output, _ = self.temporal_fusion(H_T.unsqueeze(1))
        H_T = H_T + temporal_output.squeeze(1)  # 残差连接
        
        # 多路径特征提取
        devign_vec = self.devign_conv(H_T, x0, g.batch)
        att_vec = self.hierarchical_attention(H_T, g.batch)
        
        # 特征融合
        out = torch.cat([devign_vec, att_vec], dim=1)
        return self.fuse(out)


# SARD数据集优化组件
class SARDDataOptimizer:
    """SARD数据集特定优化器"""
    def __init__(self, config: DictConfig):
        self.config = config
        self.vulnerability_patterns = self._load_sard_patterns()
        
    def _load_sard_patterns(self):
        """加载SARD特定的漏洞模式"""
        # 实现SARD数据集特定的模式加载
        return {}
    
    def augment_training_data(self, batch_data, difficulty_level: int):
        """根据难度级别增强训练数据"""
        # 实现基于SARD的数据增强
        return batch_data
    
    def filter_by_difficulty(self, batch_data, max_difficulty: int):
        """根据难度过滤样本"""
        if hasattr(batch_data, 'difficulty'):
            mask = batch_data.difficulty <= max_difficulty
            return self._apply_mask_to_batch(batch_data, mask)
        return batch_data
    
    def _apply_mask_to_batch(self, batch_data, mask):
        """将掩码应用到批次数据"""
        # 实现批次数据的掩码应用
        return batch_data


# 保留原有的编码器以保持兼容性
class GraphConvEncoder(torch.nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GCNConv(config.rnn.hidden_size, config.hidden_size)
        self.input_GPL = TopKPooling(config.hidden_size, ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}", GCNConv(config.hidden_size, config.hidden_size))
            setattr(self, f"hidden_GPL{i}", TopKPooling(config.hidden_size, ratio=config.pooling_ratio))

        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None, batch)
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            out += self.attpool(node_embedding, batch)
        return out


class GatedGraphConvEncoder(torch.nn.Module):
    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GatedGraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru)
        self.input_GPL = TopKPooling(config.hidden_size, ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}", GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru))
            setattr(self, f"hidden_GPL{i}", TopKPooling(config.hidden_size, ratio=config.pooling_ratio))
            
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None, batch)
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            out += self.attpool(node_embedding, batch)
        return out


# 向后兼容的GGRNDevignEncoder
class GGRNDevignEncoder(EnhancedGGRNDevignEncoder):
    """向后兼容的GGRNDevignEncoder"""
    pass