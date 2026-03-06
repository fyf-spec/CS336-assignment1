"""Experiment configuration dataclass.

All hyperparameters for a training run are stored here so that they can
be serialised to the experiment log (W&B / JSON) and reconstructed from
a checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    """Complete configuration for one training experiment.

    Default values correspond to the 17M-parameter TinyStories baseline
    described in the CS336 assignment.
    """

    # ── Experiment identity ──────────────────────────────────────────
    experiment_name: str = "baseline"
    tags: list[str] = field(default_factory=list)
    description: str = ""

    # ── Model ────────────────────────────────────────────────────────
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1_376          # compute_d_ff(512) => 1376
    rope_theta: float = 10_000.0

    # ── Optimizer (AdamW) ────────────────────────────────────────────
    lr: float = 1e-3
    min_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # ── LR schedule (cosine with warmup) ─────────────────────────────
    warmup_iters: int = 200
    cosine_cycle_iters: int = 5_000

    # ── Training ─────────────────────────────────────────────────────
    iterations: int = 5_000
    batch_size: int = 64
    seed: int = 42

    # ── Data paths ───────────────────────────────────────────────────
    train_data: str = ""
    val_data: str = ""

    # ── Logging & checkpointing ──────────────────────────────────────
    log_interval: int = 10       # log train metrics every N steps
    eval_interval: int = 100     # run val eval every N steps
    eval_iters: int = 20         # number of batches per val evaluation
    checkpoint_interval: int = 500
    checkpoint_dir: str = "checkpoints"
    run_dir: str = "runs"        # root dir for experiment logs

    # ── W&B ──────────────────────────────────────────────────────────
    use_wandb: bool = True
    wandb_project: str = "cs336-experiments"
    wandb_entity: Optional[str] = None

    # ── Device ───────────────────────────────────────────────────────
    device: str = "cuda"

    # ── Helpers ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a plain dict (safe for JSON / W&B config)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        """Reconstruct from a plain dict (e.g. loaded from JSON)."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def save(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def num_params_approx(self) -> int:
        """Very rough parameter count (for display only)."""
        # Embedding + lm_head + transformer blocks
        embed = self.vocab_size * self.d_model
        lm_head = self.d_model * self.vocab_size
        # Each block: attn (4 * d_model^2) + ffn (3 * d_model * d_ff) + 2 * RMSNorm (d_model)
        block = 4 * self.d_model**2 + 3 * self.d_model * self.d_ff + 2 * self.d_model
        return embed + lm_head + self.num_layers * block
