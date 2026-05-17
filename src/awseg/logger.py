from __future__ import annotations

from typing import Any, Dict, Optional


class ExperimentLogger:
    """Lightweight experiment logger with optional Weights & Biases support.

    This wrapper lets the training code call logger.log(), logger.watch(),
    and logger.finish() regardless of whether wandb is enabled.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.wandb_config = config.get("wandb", {})
        self.enabled = bool(self.wandb_config.get("enabled", False))
        self.run = None
        self._wandb = None

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self) -> None:
        try:
            import wandb
        except ImportError as exc:
            self.enabled = False
            print(
                "[Logger] wandb is enabled in config, but wandb is not installed. "
                "Run `pip install wandb` or set `wandb.enabled: false`."
            )
            return

        self._wandb = wandb

        project = self.wandb_config.get("project", "adverse-weather-segmentation")
        entity = self.wandb_config.get("entity", None)
        run_name = self.wandb_config.get("run_name", None)
        tags = self.wandb_config.get("tags", None)
        mode = self.wandb_config.get("mode", None)

        init_kwargs = {
            "project": project,
            "entity": entity,
            "name": run_name,
            "tags": tags,
            "config": self.config,
        }

        if mode is not None:
            init_kwargs["mode"] = mode

        # Remove None values because wandb.init accepts omitted args more cleanly.
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

        self.run = wandb.init(**init_kwargs)

    @property
    def log_interval(self) -> int:
        return int(self.wandb_config.get("log_interval", 20))

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log scalar values.

        Args:
            data: Dictionary of values to log.
            step: Optional global step.
        """
        if not self.enabled or self._wandb is None:
            return

        if step is None:
            self._wandb.log(data)
        else:
            self._wandb.log(data, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: str,
        step: Optional[int] = None,
    ) -> None:
        """Log metric dictionary with a prefix.

        Example:
            metrics = {"loss": 0.3, "miou": 0.5}
            prefix = "val"
            logged keys: "val/loss", "val/miou"
        """
        data = {f"{prefix}/{key}": value for key, value in metrics.items()}
        self.log(data, step=step)

    def log_class_iou(
        self,
        class_iou: list[float],
        class_names: Optional[list[str]] = None,
        prefix: str = "val_iou",
        step: Optional[int] = None,
    ) -> None:
        """Log class-wise IoU values.

        Args:
            class_iou: List of IoU values.
            class_names: Optional class names. If omitted, class indices are used.
            prefix: wandb key prefix.
            step: Optional global step.
        """
        if class_names is None:
            class_names = [f"class_{idx}" for idx in range(len(class_iou))]

        data = {
            f"{prefix}/{name}": float(iou)
            for name, iou in zip(class_names, class_iou)
        }
        self.log(data, step=step)

    def watch(self, model: Any, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch a model with wandb.

        Args:
            model: PyTorch model.
            log: One of "gradients", "parameters", "all", or None.
            log_freq: Logging frequency.
        """
        if not self.enabled or self._wandb is None:
            return

        self._wandb.watch(model, log=log, log_freq=log_freq)

    def define_metric(self, name: str, step_metric: Optional[str] = None) -> None:
        """Define a wandb metric.

        This is optional, but useful when using custom x-axis such as epoch.
        """
        if not self.enabled or self._wandb is None:
            return

        if step_metric is None:
            self._wandb.define_metric(name)
        else:
            self._wandb.define_metric(name, step_metric=step_metric)

    def finish(self) -> None:
        if not self.enabled or self._wandb is None:
            return

        self._wandb.finish()


def build_logger(config: Dict[str, Any]) -> ExperimentLogger:
    """Build experiment logger from config."""
    return ExperimentLogger(config)
