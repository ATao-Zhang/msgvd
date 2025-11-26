from torch import nn
from omegaconf import DictConfig
import torch
import numpy
from gensim.models import KeyedVectors
from src.vocabulary import Vocabulary
from os.path import exists
from typing import List, Optional
import torch.nn.functional as F


def linear_after_attn(in_dim: int, out_dim: int, activation: str) -> nn.Module:
    """Linear layers after attention with enhanced structure"""
    return torch.nn.Sequential(
        torch.nn.Linear(2 * in_dim, 2 * in_dim),
        torch.nn.BatchNorm1d(2 * in_dim),
        get_activation(activation),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(2 * in_dim, out_dim),
    )


activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "lkrelu": nn.LeakyReLU(0.3),
    "gelu": nn.GELU()
}


def get_activation(activation_name: str) -> torch.nn.Module:
    if activation_name in activations:
        return activations[activation_name]
    raise KeyError(f"Activation {activation_name} is not supported")


class MultiScaleConv1D(nn.Module):
    """多尺度1D卷积模块"""
    def __init__(self, embed_size: int, out_channels: int = 64):
        super().__init__()
        self.conv3 = nn.Conv1d(embed_size, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_size, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_size, out_channels, kernel_size=7, padding=3)
        self.attention_weights = nn.Parameter(torch.ones(3) / 3)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.residual_proj = nn.Linear(embed_size, out_channels) if embed_size != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, embed_size]
        x_permuted = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        
        conv3_out = self.conv3(x_permuted).permute(0, 2, 1)  # [batch_size, seq_len, out_channels]
        conv5_out = self.conv5(x_permuted).permute(0, 2, 1)
        conv7_out = self.conv7(x_permuted).permute(0, 2, 1)
        
        # 自适应权重融合
        weights = torch.softmax(self.attention_weights, dim=0)
        fused_output = (weights[0] * conv3_out + weights[1] * conv5_out + weights[2] * conv7_out)
        
        # 残差连接 + 层归一化
        residual = self.residual_proj(x)
        output = self.layer_norm(fused_output + residual)
        
        return output


class ExecutionPathEncoder:
    """执行路径编码器，为CFG节点提供执行顺序"""
    def __init__(self, config: DictConfig):
        self.config = config
        
    def get_execution_sequence(self, cfg_nodes: List, method_ast) -> List[int]:
        """基于控制流分析生成节点执行顺序"""
        # 简化实现：基于拓扑排序或深度优先遍历
        # 实际实现需要根据具体的CFG结构进行分析
        sequence = list(range(len(cfg_nodes)))
        return sequence


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class EnhancedRNNLayer(torch.nn.Module):
    """增强的RNN层，包含双向GRU和时序注意力"""
    def __init__(self, config: DictConfig, pad_idx: int):
        super(EnhancedRNNLayer, self).__init__()
        self.__pad_idx = pad_idx
        self.__config = config
        
        # 多尺度卷积预处理
        self.multiscale_conv = MultiScaleConv1D(config.embed_size, config.rnn.hidden_size)
        
        # 双向GRU
        self.__rnn = nn.GRU(
            input_size=config.rnn.hidden_size,
            hidden_size=config.rnn.hidden_size // 2,
            num_layers=config.rnn.num_layers,
            bidirectional=True,
            dropout=config.rnn.drop_out if config.rnn.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 时序注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.rnn.hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.__dropout_rnn = nn.Dropout(config.rnn.drop_out)
        self.position_encoding = PositionalEncoding(config.rnn.hidden_size)
        self.execution_encoder = ExecutionPathEncoder(config)

    def forward(self, subtokens_embed: torch.Tensor, node_ids: torch.Tensor, execution_sequence: Optional[List[int]] = None):
        # 多尺度卷积特征提取
        conv_features = self.multiscale_conv(subtokens_embed)
        
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(
                node_ids == self.__pad_idx, dim=1)
            first_pad_pos[~is_contain_pad_id] = node_ids.shape[1]
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos, descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
            
        conv_features = conv_features[sort_indices]
        
        # 位置编码
        conv_features = self.position_encoding(conv_features)
        
        # 如果有执行序列信息，重新排序
        if execution_sequence is not None:
            # 根据执行序列重新组织特征
            pass
        
        # GRU处理
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            conv_features, sorted_path_lengths, batch_first=True)
        rnn_output, hidden = self.__rnn(packed_embeddings)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        
        # 时序注意力
        attn_output, _ = self.temporal_attention(rnn_output, rnn_output, rnn_output)
        
        # 合并GRU输出和注意力输出
        combined_output = rnn_output + attn_output
        
        # 获取最终节点嵌入
        batch_size = combined_output.size(0)
        node_embedding = torch.zeros(batch_size, combined_output.size(-1), 
                                   device=combined_output.device)
        for i in range(batch_size):
            valid_length = sorted_path_lengths[i]
            if valid_length > 0:
                node_embedding[i] = combined_output[i, valid_length - 1]
        
        node_embedding = self.__dropout_rnn(node_embedding)[reverse_sort_indices]
        return node_embedding


class EnhancedSTEncoder(torch.nn.Module):
    """增强的语句编码器"""
    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(EnhancedSTEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__wd_embedding = nn.Embedding(vocabulary_size,
                                           config.embed_size,
                                           padding_idx=pad_idx)
        torch.nn.init.xavier_uniform_(self.__wd_embedding.weight.data)
        
        if exists(config.w2v_path):
            self.__add_w2v_weights(config.w2v_path, vocab)
            
        self.__rnn_attn = EnhancedRNNLayer(config, pad_idx)

    def __add_w2v_weights(self, w2v_path: str, vocab: Vocabulary):
        model = KeyedVectors.load(w2v_path, mmap="r")
        w2v_weights = self.__wd_embedding.weight.data
        for wd in model.index2word:
            w2v_weights[vocab.convert_token_to_id(wd)] = torch.from_numpy(model[wd])
        self.__wd_embedding.weight.data.copy_(w2v_weights)

    def forward(self, seq: torch.Tensor, execution_sequence: Optional[List[int]] = None):
        wd_embedding = self.__wd_embedding(seq)
        node_embedding = self.__rnn_attn(wd_embedding, seq, execution_sequence)
        return node_embedding


# 保持向后兼容的原始STEncoder
class STEncoder(EnhancedSTEncoder):
    """向后兼容的原始语句编码器"""
    def forward(self, seq: torch.Tensor):
        return super().forward(seq)