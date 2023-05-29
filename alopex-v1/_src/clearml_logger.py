from __future__ import annotations

from .loggers import Logger
from .pytypes import Summary, LoggerState

try:
    _CLEARML_AVAILABLE = True
    from clearml import Task
except ImportError:
    _CLEARML_AVAILABLE = False


class ClearmlLogger(Logger):
    """Clear-ml logger

    Args:
        api_key: Comet API key.
        project_name: Project name.
        experiment_name: Experiment name.
        experiment_key: Experiment key to re-use the previous experiment.
            Note that if you load state_dict of previous loggers,
            You do not have to specify this argument.
        disabled: Whether to disable logging to Comet. Useful for debugging.
    """

    def __init__(
        self,
        project_name: str | None = None,
        experiment_name: str | None = None,
        experiment_key: str | None = None,
        **kwargs,
    ) -> None:
        if not _CLEARML_AVAILABLE:
            raise ImportError(
                "Failed to import clearml. Install clearml in advance.",
            )
        self._project_name = project_name
        self._experiment_name = experiment_name
        self._experiment_key = experiment_key
        self._kwargs = kwargs
        self._experiment = None

    @property
    def experiment(self) -> Task:
        if self._experiment is None:
            self._experiment = Task.init(
                project_name=self._project_name,
                task_name=self._experiment_name,
                reuse_last_task_id=self._experiment_key or False,
                **self._kwargs,
            )
            if self._experiment_key is None:
                self._experiment_key = self._experiment.task_id
        return self._experiment

    def log_summary(self, summary: Summary, step: int | None = None, epoch: int | None = None) -> None:
        logger = self.experiment.get_logger()
        for key, value in summary.items():
            if step is not None:
                logger.report_scalar(title="summary (it)", series=key, value=value, iteration=step)
            if epoch is not None:
                logger.report_scalar(title="summary (epoch)", series=key, value=value, iteration=epoch)

    def log_hparams(self, hparams: dict) -> None:
        self.experiment.connect(hparams)

    def state_dict(self) -> LoggerState:
        return {"experiment_key": self._experiment_key}

    def load_state_dict(self, state: LoggerState) -> None:
        self._experiment_key = state["experiment_key"]
