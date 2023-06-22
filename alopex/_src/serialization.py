from __future__ import annotations
import typing as tp
from pathlib import Path
import pickle
import gzip
import json


def save_obj(filename: str | Path, obj: tp.Any) -> None:
    """Compress and save a picklable object into `filename`.

    Args:
        filename: path to save `obj`.
        obj: a picklable object to save.
    """
    file_path = Path(filename)
    content = gzip.compress(pickle.dumps(obj))
    tmp_file_path = file_path.with_suffix(".tmp")
    tmp_file_path.write_bytes(content)
    tmp_file_path.rename(file_path)


def load_obj(filename: str | Path) -> tp.Any:
    """Load an object saved by `save_obj`.

    Args:
        filename: path to the object.

    Return:
        The loaded object.
    """
    file_path = Path(filename)
    obj = pickle.loads(gzip.decompress(file_path.read_bytes()))
    return obj


def save_log(filename: str | Path, log: tp.Sequence[dict[str, tp.Any]]) -> None:
    """Write the seqence of summaries as JSON.

    Args:
        filename: a path to write `log`.
        log: a sequence of summaries.
    """
    file_path = Path(filename)
    tmp_file_path = file_path.with_suffix(".tmp")
    tmp_file_path.write_text(json.dumps(log, indent=2))
    tmp_file_path.rename(file_path)


def load_log(filename: str | Path) -> tp.Sequence[dict[str, tp.Any]]:
    """Load JSON-formatted log.

    Args:
        filename: path to load.

    Returns:
        A restored seqence of summaries.
    """
    file_path = Path(filename)
    log = file_path.read_text()
    log = json.loads(log)
    return log
