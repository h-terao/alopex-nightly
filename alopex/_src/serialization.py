from __future__ import annotations
import typing as tp
from pathlib import Path
import pickle
import gzip
import json


def save_obj(filename: str | Path, obj: tp.Any) -> None:
    file_path = Path(filename)
    content = gzip.compress(pickle.dumps(obj))
    tmp_file_path = file_path.with_suffix(".tmp")
    tmp_file_path.write_bytes(content)
    tmp_file_path.rename(file_path)


def load_obj(filename: str | Path) -> tp.Any:
    file_path = Path(filename)
    obj = pickle.loads(gzip.decompress(file_path.read_bytes()))
    return obj


def save_log(filename: str | Path, log: tp.Sequence[dict[str, tp.Any]]):
    file_path = Path(filename)
    tmp_file_path = file_path.with_suffix(".tmp")
    tmp_file_path.write_text(json.dumps(log, indent=2))
    tmp_file_path.rename(file_path)


def load_log(filename: str | Path) -> tp.Sequence[dict[str, tp.Any]]:
    file_path = Path(filename)
    log = file_path.read_text()
    log = json.loads(log)
    return log
