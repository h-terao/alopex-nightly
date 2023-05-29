"""Deep learning loggers.

Example:
    ```python
    logger = LoggerCollection(
        ConsoleLogger(),
        JsonLogger(),
        TensorBoardLogger(),
    )

    summary = {"train/loss": 0, "test/loss": 0.2}
    logger.log_summary(summary, step=1024, epoch=1)

    lg_state = logger.state_dict()  # Get logger state.
    logger.load_state_dict(lg_state)  # Restore logger.
    ```
"""
from __future__ import annotations
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path
import json
import yaml

from .pytypes import Summary, LoggerState


class Logger(ABC):
    """An abstract base class of loggers."""

    @abstractmethod
    def log_summary(
        self, summary: Summary, step: int | None = None, epoch: int | None = None
    ) -> None:
        pass

    def log_hparams(self, hparams: dict) -> None:
        pass

    def log_code(self, code_path: str | Path) -> None:
        pass

    @abstractmethod
    def state_dict(self) -> LoggerState:
        pass

    @abstractmethod
    def load_state_dict(self, state: LoggerState) -> None:
        pass


class LoggerCollection(Logger):
    """Support to use multiple loggers at the same time.

    Args:
        loggers: Loggers.
    """

    def __init__(self, *loggers) -> None:
        self._loggers: tp.Sequence[Logger] = loggers

    def log_summary(
        self, summary: Summary, step: int | None = None, epoch: int | None = None
    ) -> None:
        for lg in self._loggers:
            lg.log_summary(summary, step, epoch)

    def log_hparams(self, hparams: dict) -> None:
        for lg in self._loggers:
            lg.log_hparams(hparams)

    def log_code(self, code_path: str | Path) -> None:
        for lg in self._loggers:
            lg.log_code(code_path)

    def state_dict(self) -> LoggerState:
        return [lg.state_dict() for lg in self._loggers]

    def load_state_dict(self, state: LoggerState) -> None:
        for lg, lg_state in zip(self._loggers, state):
            lg.load_state_dict(lg_state)

    def __iter__(self):
        yield from self._loggers


class ConsoleLogger(Logger):
    """Print values on console.

    Args:
        print_fun: Function to print summary and hparams.
            If None, use the `print` function.
    """

    def __init__(self, print_fun: tp.Callable | None = None) -> None:
        self._print_fun = print_fun or print

    def log_summary(
        self, summary: Summary, step: int | None = None, epoch: int | None = None
    ) -> None:
        values = dict()
        if step is not None:
            values["step"] = step
        if epoch is not None:
            values["epoch"] = epoch
        self._print_fun(dict(values, **summary))

    def log_hparams(self, hparams: dict) -> None:
        self._print_fun(yaml.dump(hparams, allow_unicode=True))

    def state_dict(self) -> LoggerState:
        return dict()

    def load_state_dict(self, state: LoggerState) -> None:
        del state  # unused.
        return None


class DiskLogger(Logger):
    """A logger that dumps log in local disk.

    Args:
        save_dir: Directory to save files. If the specified directory does not exist,
            DiskLogger makes a directory.
        log_file_name: Filename to write the logged summary.
        hparams_file_name: Filename to write the logged hyperparams.
    """

    def __init__(
        self,
        save_dir: str | Path,
        log_file_name: str | None = "log.json",
        hparams_file_name: str | None = "hparams.yaml",
    ) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self._log_file = None if log_file_name is None else Path(save_dir, log_file_name)
        self._hparams_file = (
            None if hparams_file_name is None else Path(save_dir, hparams_file_name)
        )

        self._log = []
        self._hparams = dict()

    def log_summary(
        self, summary: Summary, step: int | None = None, epoch: int | None = None
    ) -> None:
        if self._log_file is not None:
            values = dict()
            if step is not None:
                values["step"] = step
            if epoch is not None:
                values["epoch"] = epoch
            values = dict(values, **summary)

            self._log.append(values)
            self._tmp_log_file.write_text(json.dumps(self._log, indent=2))
            self._tmp_log_file.rename(self._log_file)

    def log_hparams(self, hparams: dict) -> None:
        if self._hparams_file is not None:
            self._hparams = dict(self._hparams, **hparams)
            self._tmp_hparams_file.write_text(yaml.dump(self._hparams, allow_unicode=True))
            self._tmp_hparams_file.rename(self._hparams_file)

    def state_dict(self) -> LoggerState:
        return {"_log": self._log, "_hparams": self._hparams}

    def load_state_dict(self, state: LoggerState) -> None:
        self._log = state["_log"]
        self._hparams = state["_hparams"]

    @property
    def _tmp_log_file(self) -> Path | None:
        return None if self._log_file is None else self._log_file.with_suffix(".tmp")

    @property
    def _tmp_hparams_file(self) -> Path | None:
        return None if self._hparams_file is None else self._hparams_file.with_suffix(".tmp")
