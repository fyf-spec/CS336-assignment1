"""Training script with integrated experiment logging.

Usage:
    python -m Experiments.train \
        --train_data data/TinyStories_train.bin \
        --val_data data/TinyStories_val.bin \
        --experiment_name baseline_17M \
        --iterations 5000

    # Disable W&B (JSON-only):
    python -m Experiments.train --no_wandb ...

    # Resume from checkpoint:
    python -m Experiments.train --resume runs/baseline_17M/checkpoints/ckpt_1000.pt ...
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time

import numpy as np
import torch

from cs336_basics.nn import TransformerLM, cross_entropy_loss
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, clip_gradient_norm
from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint

from Experiments.config import ExperimentConfig
from Experiments.logger import ExperimentLogger

# ── Logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Argument parsing ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer LM with experiment logging")

    # Experiment identity
    p.add_argument("--experiment_name", type=str, default="baseline")
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--description", type=str, default="")

    # Model
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--rope_theta", type=float, default=None)

    # Optimizer
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)

    # Schedule
    p.add_argument("--warmup_iters", type=int, default=None)
    p.add_argument("--cosine_cycle_iters", type=int, default=None)

    # Training
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Data
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, required=True)

    # Logging
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_iters", type=int, default=None)
    p.add_argument("--checkpoint_interval", type=int, default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create an ExperimentConfig using defaults, overridden by CLI args."""
    cfg = ExperimentConfig()

    # Override any field that was explicitly set on the command line
    for key, value in vars(args).items():
        if key == "no_wandb":
            if value:
                cfg.use_wandb = False
            continue
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)

    # Auto-detect device if not specified
    if args.device is None:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg


# ── Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str,
) -> float:
    """Estimate mean loss over `eval_iters` random batches."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


class ActivationMonitor:
    """Helper to monitor the maximum activation norm across Linear layers."""
    def __init__(self, model: torch.nn.Module):
        self.max_norm = 0.0
        self.hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.hooks.append(module.register_forward_hook(self.hook_fn))
                
    def hook_fn(self, module, input, output):
        norm = output.detach().norm(2).item()
        if norm > self.max_norm:
            self.max_norm = norm
            
    def get_and_reset(self) -> float:
        val = self.max_norm
        self.max_norm = 0.0
        return val
        
    def remove(self):
        for h in self.hooks: h.remove()


# ── Training loop ────────────────────────────────────────────────────

def train(cfg: ExperimentConfig) -> None:
    """Full training loop with integrated experiment logging."""

    # ── Reproducibility ──────────────────────────────────────────────
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Data ─────────────────────────────────────────────────────────
    train_data = np.memmap(cfg.train_data, dtype=np.uint16, mode="r")
    val_data   = np.memmap(cfg.val_data,   dtype=np.uint16, mode="r")
    log.info("Train tokens: %s  |  Val tokens: %s", f"{len(train_data):,}", f"{len(val_data):,}")

    # ── Model ────────────────────────────────────────────────────────
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=torch.device(cfg.device),
    ).to(cfg.device)

    num_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{num_params:,}")

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # ── Resume ───────────────────────────────────────────────────────
    start_iter = 0
    resume_path = getattr(cfg, "_resume_path", None)
    if resume_path:
        start_iter = load_checkpoint(resume_path, model, optimizer)
        log.info("Resumed from checkpoint %s at iter %d", resume_path, start_iter)

    # ── Logger ───────────────────────────────────────────────────────
    exp_logger = ExperimentLogger(cfg)
    exp_logger.log_config(extra={"num_params": num_params})

    # ── Checkpoint dir (under the run directory) ─────────────────────
    ckpt_dir = exp_logger.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Initial validation ───────────────────────────────────────────
    val_loss = estimate_loss(model, val_data, cfg.batch_size, cfg.context_length, cfg.eval_iters, cfg.device)
    log.info("Initial val loss: %.4f  |  ppl: %.2f", val_loss, math.exp(val_loss))
    exp_logger.log_eval(step=0, metrics={"val/loss": val_loss, "val/ppl": math.exp(val_loss)})

    # ── Activation Monitor ───────────────────────────────────────────
    act_monitor = ActivationMonitor(model)

    # ── Training loop ────────────────────────────────────────────────
    model.train()
    tokens_per_step = cfg.batch_size * cfg.context_length
    t_step_start = time.time()

    for it in range(start_iter, cfg.iterations):
        # ── Learning-rate schedule ───────────────────────────────────
        lr = get_lr_cosine_schedule(
            it,
            max_learning_rate=cfg.lr,
            min_learning_rate=cfg.min_lr,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.cosine_cycle_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Forward / backward ───────────────────────────────────────
        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping (log norm before and after)
        grad_norm_before = _grad_norm(model)
        if cfg.grad_clip > 0:
            clip_gradient_norm(model.parameters(), cfg.grad_clip)
        grad_norm_after = _grad_norm(model)

        optimizer.step()

        # ── Per-step logging ─────────────────────────────────────────
        if (it + 1) % cfg.log_interval == 0:
            dt = time.time() - t_step_start
            tokens_per_sec = tokens_per_step * cfg.log_interval / dt

            step_metrics = {
                "train/loss": loss.item(),
                "lr": lr,
                "weight_norm": _weight_norm(model),
                "activation_norm": act_monitor.get_and_reset(),
                "grad_norm_before_clip": grad_norm_before,
                "grad_norm_after_clip": grad_norm_after,
                "tokens_per_sec": tokens_per_sec,
                "step_time_ms": dt / cfg.log_interval * 1000,
            }
            exp_logger.log_step(step=it + 1, metrics=step_metrics)

            log.info(
                "step %5d/%d | loss %.4f | lr %.2e | tok/s %s | %.1f ms/step",
                it + 1, cfg.iterations, loss.item(), lr,
                f"{tokens_per_sec:,.0f}", dt / cfg.log_interval * 1000,
            )
            t_step_start = time.time()

        # ── Periodic evaluation ──────────────────────────────────────
        if (it + 1) % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, cfg.batch_size, cfg.context_length, cfg.eval_iters, cfg.device)
            val_ppl = math.exp(val_loss)
            exp_logger.log_eval(step=it + 1, metrics={"val/loss": val_loss, "val/ppl": val_ppl})
            log.info("  ↳ val loss %.4f | ppl %.2f", val_loss, val_ppl)
            model.train()
            t_step_start = time.time()  # don't count eval time

        # ── Checkpointing ────────────────────────────────────────────
        if (it + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_{it + 1}.pt"
            save_checkpoint(model, optimizer, it + 1, str(ckpt_path))
            log.info("  ↳ saved checkpoint → %s", ckpt_path)

        # ── Periodic local save (guard against crashes) ──────────────
        if (it + 1) % (cfg.eval_interval * 5) == 0:
            exp_logger.save_local()

    # ── Final checkpoint + clean up ──────────────────────────────────
    final_path = ckpt_dir / "ckpt_final.pt"
    save_checkpoint(model, optimizer, cfg.iterations, str(final_path))
    log.info("Final checkpoint → %s", final_path)
    exp_logger.finish()
    log.info("Training complete.")


def _weight_norm(model: torch.nn.Module) -> float:
    """Compute the total L2 norm of all parameters."""
    return torch.sqrt(sum(torch.sum(p ** 2) for p in model.parameters())).item()

def _grad_norm(model: torch.nn.Module) -> float:
    """Compute the total L2 norm of all gradients."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.sqrt(sum(torch.sum(g ** 2) for g in grads)).item()


# ── Entry point ──────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # Stash resume path (not part of the config dataclass)
    if args.resume:
        cfg._resume_path = args.resume  # type: ignore[attr-defined]

    log.info("Experiment: %s", cfg.experiment_name)
    log.info("Device: %s", cfg.device)
    log.info("Approx params: %s", f"{cfg.num_params_approx:,}")

    train(cfg)


if __name__ == "__main__":
    main()
