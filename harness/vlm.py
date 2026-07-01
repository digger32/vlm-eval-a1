"""VLM facade: a thin wrapper over adapters with a per-item soft timeout and a
uniform ask() signature used by both drivers."""
from __future__ import annotations
import signal
import torch
from contextlib import contextmanager

from . import adapters


class ItemTimeout(Exception):
    pass


@contextmanager
def _soft_timeout(seconds: int):
    """SIGALRM-based soft timeout for a single generate() call. Catches Python-level
    hangs; truly wedged CUDA kernels are caught by the orchestrator's HARD per-unit
    subprocess timeout instead (see runner/orchestrate.py)."""
    if not seconds:
        yield
        return

    def _handler(signum, frame):
        raise ItemTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class VLM:
    def __init__(self, spec, gen_defaults: dict, item_timeout_s: int = 120):
        self.spec = spec
        self.handle = adapters.load(spec)
        self.gen_kwargs = {
            "max_new_tokens": gen_defaults.get("max_new_tokens", 64),
            "do_sample": gen_defaults.get("do_sample", False),
        }
        self.item_timeout_s = item_timeout_s

    @torch.inference_mode()
    def ask(self, image, prompt: str) -> str:
        with _soft_timeout(self.item_timeout_s):
            return adapters.ask(self.handle, image, prompt, self.gen_kwargs)

    def peak_vram_bytes(self) -> int:
        return int(torch.cuda.max_memory_allocated())
