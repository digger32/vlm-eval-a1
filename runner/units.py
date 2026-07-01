"""Enumerate units from a run config.

A unit = (paper, model, dataset, condition, seed). The A2 'perm_rand' condition with
n_perms expands into perm_rand_0..perm_rand_{n-1} so each permutation is its own unit
(independent, resumable, timeout-isolated)."""
from __future__ import annotations
import yaml
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Unit:
    paper: str
    model: str
    dataset: str
    condition: str
    seed: int

    @property
    def uid(self) -> str:
        return f"{self.paper}__{self.model}__{self.dataset}__{self.condition}__seed{self.seed}"


def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _expand_conditions(cfg: dict, cond_name: str) -> list[str]:
    spec = cfg["conditions"][cond_name]
    if spec.get("permutation") == "random":
        n = spec.get("n_perms", 1)
        return [f"{cond_name}_{i}" for i in range(n)]
    return [cond_name]


def enumerate_units(cfg: dict) -> list[Unit]:
    paper = cfg["paper"]
    seeds = cfg.get("seeds", [0])
    units: list[Unit] = []
    for row in cfg["matrix"]:
        ds = row["dataset"]
        for cond_name in row["conditions"]:
            for cond in _expand_conditions(cfg, cond_name):
                for model in cfg["models"]:
                    for seed in seeds:
                        units.append(Unit(paper, model, ds, cond, seed))
    return units


def condition_base(condition: str) -> str:
    """perm_rand_3 -> perm_rand ; blur_s3 -> blur_s3 (only strips trailing _<int> for
    expanded random perms)."""
    parts = condition.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0].endswith("perm_rand"):
        return parts[0]
    return condition
