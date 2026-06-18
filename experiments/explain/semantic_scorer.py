import re
from typing import Dict, List


DANGEROUS_APIS = (
    "strcpy",
    "wcscpy",
    "strcat",
    "wcscat",
    "sprintf",
    "vsprintf",
    "gets",
    "memcpy",
    "memmove",
    "strncpy",
    "wcsncpy",
)

MEMORY_APIS = (
    "malloc",
    "calloc",
    "realloc",
    "free",
    "alloca",
    "new",
    "delete",
)

INPUT_APIS = (
    "scanf",
    "fscanf",
    "sscanf",
    "fgets",
    "fread",
    "read",
    "recv",
    "getenv",
    "argv",
    "atoi",
    "atol",
    "strtol",
)

LENGTH_OR_BOUNDARY = (
    "strlen",
    "wcslen",
    "sizeof",
)

WEIGHTS = {
    "dangerous_api": 1.0,
    "memory_api": 0.7,
    "input_api": 0.6,
    "array_or_pointer": 0.5,
    "length_or_boundary": 0.4,
    "arithmetic": 0.2,
}


def _contains_call(code: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\s*\(", code))


def _contains_token(code: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", code))


def _pointer_like_assignment(code: str) -> bool:
    return bool(re.search(r"\*\s*\w+\s*=", code) or re.search(r"\w+\s*=\s*&\s*\w+", code))


def _has_arithmetic(code: str) -> bool:
    return bool(re.search(r"(<<|>>|[+\-*/%])", code))


def semantic_risk_score(code: str) -> Dict:
    """Return clipped semantic risk score and matched risk tags for one line."""
    text = code or ""
    tags: List[str] = []
    score = 0.0

    for api in DANGEROUS_APIS:
        if _contains_call(text, api):
            tags.append(f"dangerous_api:{api}")
    if any(tag.startswith("dangerous_api:") for tag in tags):
        score += WEIGHTS["dangerous_api"]

    for api in MEMORY_APIS:
        if _contains_call(text, api) or _contains_token(text, api):
            tags.append(f"memory_api:{api}")
    if any(tag.startswith("memory_api:") for tag in tags):
        score += WEIGHTS["memory_api"]

    for api in INPUT_APIS:
        if _contains_call(text, api) or _contains_token(text, api):
            tags.append(f"input_api:{api}")
    if any(tag.startswith("input_api:") for tag in tags):
        score += WEIGHTS["input_api"]

    if "[" in text and "]" in text:
        tags.append("array_access")
    if "->" in text or _pointer_like_assignment(text):
        tags.append("pointer_access")
    if "array_access" in tags or "pointer_access" in tags:
        score += WEIGHTS["array_or_pointer"]

    for api in LENGTH_OR_BOUNDARY:
        if _contains_call(text, api) or _contains_token(text, api):
            tags.append(f"length_check:{api}")
    if any(tag.startswith("length_check:") for tag in tags):
        score += WEIGHTS["length_or_boundary"]

    if _has_arithmetic(text):
        tags.append("arithmetic")
        score += WEIGHTS["arithmetic"]

    return {
        "semantic_score": max(0.0, min(1.0, score)),
        "semantic_tags": tags,
    }


def normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        if max_value == 0:
            return [0.0 for _ in values]
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def combine_scores(attention_score: float, semantic_score: float, mode: str, semantic_weight: float) -> float:
    semantic_weight = max(0.0, min(1.0, semantic_weight))
    if mode == "attention":
        return attention_score
    if mode == "semantic":
        return semantic_score
    if mode == "attention_semantic":
        return (1.0 - semantic_weight) * attention_score + semantic_weight * semantic_score
    raise ValueError(f"Unsupported score_mode: {mode}")
