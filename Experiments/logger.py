"""Experiment logger with W&B + local JSON dual backend.

Usage:
    from Experiments.config import ExperimentConfig
    from Experiments.logger import ExperimentLogger

    cfg = ExperimentConfig(experiment_name="my_run")
    logger = ExperimentLogger(cfg)
    logger.log_step(step=1, metrics={"train/loss": 3.14, "lr": 1e-3})
    logger.log_eval(step=100, metrics={"val/loss": 2.50, "val/ppl": 12.2})
    logger.finish()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig

log = logging.getLogger(__name__)


class ExperimentLogger:
    """Track experiment metrics to W&B and/or a local JSON file.

    Every logged metric automatically gets a ``wall_time_s`` field
    recording seconds since the logger was created (i.e. training start).
    """

    def __init__(self, config: "ExperimentConfig") -> None:
        self.config = config
        self._start_time = time.time()

        # ── Metrics storage (always kept locally) ────────────────────
        self._train_metrics: list[dict[str, Any]] = []
        self._eval_metrics: list[dict[str, Any]] = []

        # ── Run directory ────────────────────────────────────────────
        self.run_dir = Path(config.run_dir) / config.experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config alongside the metrics
        config.save(self.run_dir / "config.json")

        # ── W&B (optional) ───────────────────────────────────────────
        self._wandb_run = None
        if config.use_wandb:
            self._init_wandb(config)

        log.info(
            "ExperimentLogger initialised  |  run_dir=%s  |  wandb=%s",
            self.run_dir,
            "ON" if self._wandb_run else "OFF",
        )

    # ── Public API ───────────────────────────────────────────────────

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """Log per-step training metrics (loss, lr, grad norm, …)."""
        entry = {
            "step": step,
            "wall_time_s": self._elapsed(),
            **metrics,
        }
        self._train_metrics.append(entry)

        if self._wandb_run is not None:
            self._wandb_run.log(entry, step=step)

    def log_eval(self, step: int, metrics: dict[str, Any]) -> None:
        """Log periodic evaluation metrics (val loss, perplexity, …)."""
        entry = {
            "step": step,
            "wall_time_s": self._elapsed(),
            **metrics,
        }
        self._eval_metrics.append(entry)

        if self._wandb_run is not None:
            self._wandb_run.log(entry, step=step)

    def log_config(self, extra: dict[str, Any] | None = None) -> None:
        """Log the full config to W&B (called once at the start)."""
        if self._wandb_run is not None:
            cfg_dict = self.config.to_dict()
            if extra:
                cfg_dict.update(extra)
            self._wandb_run.config.update(cfg_dict)

    def save_local(self) -> None:
        """Flush all accumulated metrics to a local JSON file."""
        out = {
            "config": self.config.to_dict(),
            "train_metrics": self._train_metrics,
            "eval_metrics": self._eval_metrics,
        }
        path = self.run_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log.info("Saved local metrics → %s", path)

    def finish(self) -> None:
        """Finalise: flush local file and close W&B run."""
        self.save_local()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        log.info("ExperimentLogger finished.")

    # ── Convenience properties ───────────────────────────────────────

    @property
    def latest_train_loss(self) -> float | None:
        """Most recent training loss, or None if nothing logged yet."""
        if not self._train_metrics:
            return None
        return self._train_metrics[-1].get("train/loss")

    @property
    def latest_val_loss(self) -> float | None:
        """Most recent validation loss."""
        if not self._eval_metrics:
            return None
        return self._eval_metrics[-1].get("val/loss")

    # ── Private helpers ──────────────────────────────────────────────

    def _elapsed(self) -> float:
        """Seconds since logger creation (i.e. training start)."""
        return time.time() - self._start_time

    def _init_wandb(self, config: "ExperimentConfig") -> None:
        """Try to initialise W&B; fall back silently on failure."""
        try:
            import wandb  # noqa: F811

            self._wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.experiment_name,
                tags=config.tags,
                config=config.to_dict(),
                reinit=True,
            )
        except Exception as exc:
            log.warning("W&B init failed (%s) — falling back to JSON-only logging.", exc)
            self._wandb_run = None
