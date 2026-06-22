from dataclasses import dataclass
import networkx as nx
from os.path import exists
from typing import Dict, Iterable, List, Set

import torch
from torch_geometric.data import Data

from src.vocabulary import Vocabulary


_DANGEROUS_APIS = {
    "strcpy", "strncpy", "strcat", "sprintf", "vsprintf", "scanf",
    "sscanf", "gets", "memcpy", "memmove", "malloc", "calloc", "realloc",
    "free", "delete", "new", "system", "exec", "popen", "read", "recv",
    "recvfrom", "write", "send",
}


@dataclass(frozen=True)
class XFGNode:
    ln: int
    line_id: int
    file_id: str = ""
    func_id: str = ""
    stmt_type: str = ""
    code_text: str = ""


@dataclass
class XFGEdge:
    from_node: XFGNode
    to_node: XFGNode
    edge_type: str = ""


@dataclass
class XFG:
    def __init__(self, path: str = None, xfg: nx.DiGraph = None):
        if xfg is not None:
            xfg_nx: nx.DiGraph = xfg
        elif path is not None:
            assert exists(path), f"xfg {path} not exists!"
            xfg_nx: nx.DiGraph = nx.read_gpickle(path)
        else:
            raise ValueError("invalid inputs!")
        self.__init_graph(xfg_nx)

    @staticmethod
    def __first_graph_value(graph: nx.DiGraph, keys: Iterable[str], default=""):
        for key in keys:
            if key in graph.graph:
                value = graph.graph[key]
                if isinstance(value, list):
                    return value[0] if value else default
                return value
        return default

    @staticmethod
    def __node_value(xfg_nx: nx.DiGraph, node, keys: Iterable[str], default=""):
        attrs = xfg_nx.nodes[node]
        for key in keys:
            if key in attrs and attrs[key] is not None:
                return attrs[key]
        return default

    @staticmethod
    def __line_labels(xfg_nx: nx.DiGraph) -> Set[int]:
        raw_lines = set()
        for key in ("vul_lines", "vulnerable_lines", "loc_lines", "line_labels"):
            if key in xfg_nx.graph:
                raw_value = xfg_nx.graph[key]
                raw_lines.update(raw_value if isinstance(raw_value, list) else [raw_value])
        cleaned_lines = set()
        for line in raw_lines:
            try:
                cleaned_lines.add(int(line))
            except (TypeError, ValueError):
                continue
        return cleaned_lines

    @staticmethod
    def __stmt_features(tokens: List[str], stmt_type: str) -> List[float]:
        token_set = set(tokens)
        joined = " ".join(tokens)
        has_dangerous_api = any(api in token_set or api in joined for api in _DANGEROUS_APIS)
        has_pointer = "*" in joined or "->" in joined
        has_array = "[" in joined and "]" in joined
        has_check = any(tok in token_set for tok in ("if", "assert", "return"))
        has_release = any(tok in token_set for tok in ("free", "delete", "close"))
        is_call = "call" in stmt_type.lower() or "(" in joined
        return [
            float(has_dangerous_api),
            float(has_pointer),
            float(has_array),
            float(has_check),
            float(has_release),
            float(is_call),
        ]

    def __init_graph(self, xfg_nx: nx.DiGraph):
        self.__nodes, self.__edges, self.__tokens_list = [], [], []
        self.__node_to_idx: Dict[XFGNode, int] = {}
        self.__stmt_feature_list, self.__node_vul_labels = [], []
        k_to_nodes = {}
        file_id = str(self.__first_graph_value(xfg_nx, ("file_id", "file_paths", "file_path"), ""))
        func_id = str(self.__first_graph_value(xfg_nx, ("func_id", "function", "function_name"), ""))
        vulnerable_lines = self.__line_labels(xfg_nx)
        for idx, n in enumerate(xfg_nx):
            tokens = xfg_nx.nodes[n]["code_sym_token"]
            line_id = int(self.__node_value(xfg_nx, n, ("line_id", "line", "lineno"), n))
            stmt_type = str(self.__node_value(xfg_nx, n, ("stmt_type", "type", "label"), ""))
            code_text = str(self.__node_value(xfg_nx, n, ("code_text", "code", "source"), ""))
            xfg_node = XFGNode(
                ln=int(n),
                line_id=line_id,
                file_id=file_id,
                func_id=func_id,
                stmt_type=stmt_type,
                code_text=code_text,
            )
            self.__tokens_list.append(tokens)
            self.__stmt_feature_list.append(self.__stmt_features(tokens, stmt_type))
            self.__node_vul_labels.append(1 if line_id in vulnerable_lines else 0)
            self.__nodes.append(xfg_node)
            k_to_nodes[n] = xfg_node
            self.__node_to_idx[xfg_node] = idx
        for n in xfg_nx:
            for k in xfg_nx[n]:
                edge_type = xfg_nx[n][k].get("c/d", "")
                if edge_type in {"c", "d"}:
                    self.__edges.append(
                        XFGEdge(from_node=k_to_nodes[n],
                                to_node=k_to_nodes[k],
                                edge_type=edge_type))
        self.__label = xfg_nx.graph["label"]

    @property
    def nodes(self) -> List[XFGNode]:
        return self.__nodes

    @property
    def edges(self) -> List[XFGEdge]:
        return self.__edges

    @property
    def label(self) -> int:
        return self.__label

    def to_torch(self, vocab: Vocabulary, max_len: int) -> Data:
        """Convert this graph into torch-geometric graph with localization metadata.

        Besides the token matrix used by MSAVD, the returned graph stores
        node-to-source-line metadata that EP-LocNet uses to align execution-path
        evidence with source statements.
        """
        node_tokens = []
        for idx, _ in enumerate(self.nodes):
            node_tokens.append(self.__tokens_list[idx])
        node_ids = torch.full((len(node_tokens), max_len),
                              vocab.get_pad_id(),
                              dtype=torch.long)
        for tokens_idx, tokens in enumerate(node_tokens):
            ids = vocab.convert_tokens_to_ids(tokens)
            less_len = min(max_len, len(ids))
            node_ids[tokens_idx, :less_len] = torch.tensor(ids[:less_len],
                                                           dtype=torch.long)
        edge_pairs = [[self.__node_to_idx[e.from_node], self.__node_to_idx[e.to_node]]
                      for e in self.edges]
        edge_index = torch.tensor(list(zip(*edge_pairs)), dtype=torch.long) if edge_pairs else torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.tensor([0 if e.edge_type == "c" else 1 for e in self.edges], dtype=torch.long)
        line_ids = torch.tensor([node.line_id for node in self.nodes], dtype=torch.long)
        stmt_features = torch.tensor(self.__stmt_feature_list, dtype=torch.float)
        node_vul_labels = torch.tensor(self.__node_vul_labels, dtype=torch.float)

        data = Data(x=node_ids, edge_index=edge_index)
        data.edge_type = edge_type
        data.line_ids = line_ids
        data.stmt_features = stmt_features
        data.node_vul_labels = node_vul_labels
        return data
