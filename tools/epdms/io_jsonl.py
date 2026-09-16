"""Streaming JSONL I/O with atomic writing and checkpointing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if s:
                try:
                    yield json.loads(s)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON at {path}:{line_num} - {e}")


def read_jsonl_indexed(path: Path, key_fn: Callable[[Dict[str, Any]], str]) -> Dict[str, Dict[str, Any]]:
    mapping = {}
    for item in iter_jsonl(path):
        k = key_fn(item)
        mapping[k] = item
    return mapping


class AtomicJsonlWriter:
    def __init__(self, target_path: Path, append_if_exists: bool = True):
        self.target_path = Path(target_path)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.target_path.with_suffix(self.target_path.suffix + ".tmp")
        self.append = append_if_exists

        # If resume append mode, if target exists, we can append directly or use temp
        if self.append and self.target_path.exists():
            self._file = self.target_path.open("a", encoding="utf-8")
            self._is_temp = False
        else:
            self._file = self.temp_path.open("w", encoding="utf-8")
            self._is_temp = True

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        if self._is_temp and self.temp_path.exists():
            # Replace target with temp
            if self.target_path.exists():
                os.remove(self.target_path)
            os.rename(self.temp_path, self.target_path)

    def __enter__(self) -> AtomicJsonlWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
