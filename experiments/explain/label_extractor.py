import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


LABEL_KEYWORDS = (
    "flaw",
    "bad",
    "sink",
    "CWE",
    "POTENTIAL FLAW",
    "FLAW",
)

DANGEROUS_APIS = (
    "gets",
    "strcpy",
    "strcat",
    "sprintf",
    "vsprintf",
    "scanf",
    "sscanf",
    "memcpy",
    "memmove",
    "read",
    "recv",
)


@dataclass
class LabelExtractionReport:
    total_samples: int = 0
    vulnerable_samples: int = 0
    samples_with_line_labels: int = 0
    samples_without_line_labels: int = 0
    total_vul_lines: int = 0
    failures: List[Dict] = field(default_factory=list)

    def update(self, sample_id: str, label: int, true_vul_lines: Sequence[int], reason: str = ""):
        self.total_samples += 1
        if int(label) == 1:
            self.vulnerable_samples += 1
            if true_vul_lines:
                self.samples_with_line_labels += 1
                self.total_vul_lines += len(true_vul_lines)
            else:
                self.samples_without_line_labels += 1
                self.failures.append({"sample_id": sample_id, "reason": reason or "no_line_label_found"})

    def to_dict(self) -> Dict:
        avg = 0.0
        if self.samples_with_line_labels:
            avg = self.total_vul_lines / self.samples_with_line_labels
        return {
            "total_samples": self.total_samples,
            "vulnerable_samples": self.vulnerable_samples,
            "samples_with_line_labels": self.samples_with_line_labels,
            "samples_without_line_labels": self.samples_without_line_labels,
            "avg_vul_lines_per_sample": avg,
            "failures": self.failures,
        }


def read_source_lines(file_path: str) -> List[str]:
    if not file_path:
        return []
    path = Path(file_path)
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def graph_file_path(graph: Any) -> str:
    for key in ("file_path", "file_paths", "source_file", "file"):
        value = graph.graph.get(key)
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value:
            return str(value)
    return ""


def explicit_graph_line_labels(graph: Any) -> List[int]:
    raw_lines: Set[int] = set()
    for key in ("vul_lines", "vulnerable_lines", "loc_lines", "line_labels"):
        value = graph.graph.get(key)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for line in values:
            try:
                raw_lines.add(int(line))
            except (TypeError, ValueError):
                continue
    return sorted(raw_lines)


def source_line_map(graph: Any, source_lines: Optional[Sequence[str]] = None) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for node in graph:
        attrs = graph.nodes[node]
        line_no = attrs.get("line_id", attrs.get("line", attrs.get("lineno", node)))
        try:
            line_no = int(line_no)
        except (TypeError, ValueError):
            continue
        code = attrs.get("code_text", attrs.get("code", attrs.get("source", "")))
        if not code and source_lines and 1 <= line_no <= len(source_lines):
            code = source_lines[line_no - 1]
        mapping[line_no] = str(code)
    return mapping


def _keyword_lines(lines: Sequence[str]) -> List[int]:
    hits = []
    for idx, line in enumerate(lines, start=1):
        if any(keyword.lower() in line.lower() for keyword in LABEL_KEYWORDS):
            hits.append(idx)
    return hits


def _dangerous_api_lines(lines: Sequence[str], candidate_lines: Optional[Iterable[int]] = None) -> List[int]:
    candidates = set(int(line) for line in candidate_lines) if candidate_lines else None
    hits = []
    in_bad_function = False
    brace_depth = 0
    func_pattern = re.compile(r"\b\w*bad\w*\s*\(")
    api_pattern = re.compile(r"\b(" + "|".join(re.escape(api) for api in DANGEROUS_APIS) + r")\s*\(")

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if func_pattern.search(stripped):
            in_bad_function = True
            brace_depth = 0
        if in_bad_function:
            brace_depth += stripped.count("{") - stripped.count("}")
            if api_pattern.search(stripped) and (candidates is None or idx in candidates):
                hits.append(idx)
            if brace_depth <= 0 and "}" in stripped:
                in_bad_function = False
    return hits


def extract_vulnerable_lines(
    graph: Any,
    source_lines: Optional[Sequence[str]] = None,
    prefer_explicit: bool = False,
) -> Tuple[List[int], str]:
    """Extract SARD/Juliet weak line labels for one graph.

    The minimal experiment defaults to keyword/API weak labels. Explicit graph
    labels can be preferred by passing ``prefer_explicit=True``.
    """
    explicit = explicit_graph_line_labels(graph)
    if prefer_explicit and explicit:
        return explicit, "graph_metadata"

    if source_lines is None:
        source_lines = read_source_lines(graph_file_path(graph))

    keyword_hits = _keyword_lines(source_lines)
    if keyword_hits:
        return sorted(set(keyword_hits)), "keyword_comment"

    candidate_lines = []
    for node in graph:
        try:
            candidate_lines.append(int(graph.nodes[node].get("line_id", node)))
        except (TypeError, ValueError):
            continue
    api_hits = _dangerous_api_lines(source_lines, candidate_lines)
    if api_hits:
        return sorted(set(api_hits)), "bad_function_dangerous_api"

    if explicit:
        return explicit, "graph_metadata_fallback"
    return [], "not_found"
