from __future__ import annotations
from pathlib import Path

from .loggers import Logger
from .pytypes import Summary, LoggerState

try:
    _COMET_AVAILABLE = True
    from comet_ml import Experiment, ExistingExperiment
except ImportError:
    _COMET_AVAILABLE = False


class CometLogger(Logger):
    """Comet-ml logger

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
        api_key: str | None = None,
        project_name: str | None = None,
        experiment_name: str | None = None,
        experiment_key: str | None = None,
        **kwargs,
    ) -> None:
        if not _COMET_AVAILABLE:
            raise ImportError(
                "Failed to import comet_ml. Install comet_ml in advance.",
            )
        self._api_key = api_key
        self._project_name = project_name
        self._experiment_name = experiment_name
        self._experiment_key = experiment_key
        self._kwargs = kwargs
        self._experiment = None

    @property
    def experiment(self) -> Experiment:
        if self._experiment is None:
            if self._experiment_key is None:
                self._experiment = Experiment(
                    api_key=self._api_key,
                    project_name=self._project_name,
                    **self._kwargs,
                )
                self._experiment.set_name(self._experiment_name)
                self._experiment_key = self._experiment.get_key()
            else:
                self._experiment = ExistingExperiment(
                    api_key=self._api_key,
                    project_name=self._project_name,
                    previous_experiment=self._experiment_key,
                    **self._kwargs,
                )
        return self._experiment

    def log_summary(
        self, summary: Summary, step: int | None = None, epoch: int | None = None
    ) -> None:
        self.experiment.log_metrics(summary, step=step, epoch=epoch)

    def log_hparams(self, hparams: dict) -> None:
        self.experiment.log_parameters(hparams)

    def log_code(self, code_path: str | Path) -> None:
        self.experiment.log_code(code_path)

    def state_dict(self) -> LoggerState:
        return {"experiment_key": self._experiment_key}

    def load_state_dict(self, state: LoggerState) -> None:
        self._experiment_key = state["experiment_key"]
