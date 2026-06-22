import argparse
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEPENDENCIES = (
    "torch",
    "torch_geometric",
    "networkx",
    "gensim",
    "pandas",
    "numpy",
    "sklearn",
    "omegaconf",
    "pytorch_lightning",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check server environment for minimal explanation experiments.")
    parser.add_argument("--data_json", required=True, help="Path to split JSON, e.g. /server/path/to/SARD/test.json.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained MSAVD checkpoint.")
    parser.add_argument("--w2v", default=None, help="Path to word2vec .wv file.")
    parser.add_argument("--vocab", default=None, help="Path to fallback pickled vocabulary.")
    parser.add_argument("--output", required=True, help="Path to write env_check_report.json.")
    return parser.parse_args()


def check_imports() -> Dict[str, Dict[str, Optional[str]]]:
    results = {}
    for module_name in DEPENDENCIES:
        try:
            module = importlib.import_module(module_name)
            results[module_name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
                "error": None,
            }
        except Exception as exc:
            results[module_name] = {
                "ok": False,
                "version": None,
                "error": repr(exc),
            }
    return results


def sample_path(sample) -> Optional[str]:
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict):
        for key in ("xfg_path", "graph_path", "gpickle_path", "path", "file_path"):
            if sample.get(key):
                return str(sample[key])
    return None


def resolve_path(path_value: Optional[str], data_json_path: Path) -> Optional[str]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    candidates = [
        Path.cwd() / path_value,
        data_json_path.parent / path_value,
        Path(__file__).resolve().parents[2] / path_value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(path)


def describe_sample(sample) -> Dict:
    if isinstance(sample, dict):
        return {"type": "dict", "fields": sorted(sample.keys())}
    return {"type": type(sample).__name__, "fields": []}


def inspect_data_json(data_json: Path) -> Dict:
    result = {
        "exists": data_json.exists(),
        "num_samples": None,
        "first_3_sample_fields": [],
        "first_3_gpickle_paths": [],
        "error": None,
    }
    if not data_json.exists():
        return result
    try:
        samples = json.loads(data_json.read_text(encoding="utf-8"))
        result["num_samples"] = len(samples) if hasattr(samples, "__len__") else None
        first_three = list(samples[:3]) if isinstance(samples, list) else []
        result["first_3_sample_fields"] = [describe_sample(sample) for sample in first_three]
        for sample in first_three:
            raw_path = sample_path(sample)
            resolved = resolve_path(raw_path, data_json)
            result["first_3_gpickle_paths"].append({
                "raw": raw_path,
                "resolved": resolved,
                "exists": bool(resolved and Path(resolved).exists()),
            })
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main():
    args = parse_args()
    data_json = Path(args.data_json)
    checkpoint = Path(args.checkpoint)
    w2v = Path(args.w2v) if args.w2v else None
    vocab = Path(args.vocab) if args.vocab else None

    report = {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": check_imports(),
        "paths": {
            "data_json": {"path": str(data_json), "exists": data_json.exists()},
            "checkpoint": {"path": str(checkpoint), "exists": checkpoint.exists()},
            "w2v": {"path": str(w2v) if w2v else None, "exists": bool(w2v and w2v.exists())},
            "vocab": {"path": str(vocab) if vocab else None, "exists": bool(vocab and vocab.exists())},
            "has_w2v_or_vocab": bool((w2v and w2v.exists()) or (vocab and vocab.exists())),
        },
        "data_json_inspection": inspect_data_json(data_json),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
