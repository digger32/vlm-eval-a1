"""Model registry: read configs/models.yaml -> ModelSpec objects."""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    adapter: str
    dtype: str = "float16"
    trust_remote_code: bool = False
    gated: bool = False
    generation: str = ""
    lineage: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def load_models(path: str) -> dict[str, ModelSpec]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out: dict[str, ModelSpec] = {}
    for name, m in cfg["models"].items():
        out[name] = ModelSpec(
            name=name,
            hf_id=m["hf_id"],
            adapter=m["adapter"],
            dtype=m.get("dtype", "float16"),
            trust_remote_code=m.get("trust_remote_code", False),
            gated=m.get("gated", False),
            generation=str(m.get("generation", "")),
            lineage=m.get("lineage", ""),
            notes=m.get("notes", ""),
            extra={k: v for k, v in m.items()
                   if k not in {"hf_id", "adapter", "dtype", "trust_remote_code",
                                "gated", "generation", "lineage", "notes"}},
        )
    return out


def load_generation_defaults(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("generation_defaults", {})
