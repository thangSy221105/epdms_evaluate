"""Streaming JSONL I/O with atomic writing, checkpoint recovery, and strict validation."""

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


def read_jsonl_indexed(
    path: Path,
    key_fn: Callable[[Dict[str, Any]], str],
    allow_duplicates: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Reads a JSONL file and indexes rows by key.
    
    If allow_duplicates is False and a duplicate key is encountered, raises ValueError.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    duplicates: List[str] = []
    for item in iter_jsonl(path):
        k = key_fn(item)
        if k in mapping:
            duplicates.append(k)
            if not allow_duplicates:
                raise ValueError(f"Duplicate key '{k}' found in {path}")
        mapping[k] = item
    return mapping


def repair_truncated_jsonl(file_path: Path) -> None:
    """If the last line of a JSONL file was partially written, truncates back to last newline."""
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return

    with file_path.open("rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Read back until newline
        pos = size - 1
        f.seek(pos)
        last_char = f.read(1)
        if last_char == b"\n":
            return  # Cleanly terminated

        # Search backward for the previous newline
        while pos > 0:
            pos -= 1
            f.seek(pos)
            if f.read(1) == b"\n":
                f.seek(pos + 1)
                f.truncate()
                return

        # If no newline found, the whole file was partial
        f.seek(0)
        f.truncate()


class AtomicJsonlWriter:
    """Crash-safe JSONL writer with atomic rename, checkpoint recovery, and NaN prevention."""

    @classmethod
    def prepare_file_for_resume(cls, target_path: Path) -> None:
        """Recovers any pending .tmp file and repairs truncated trailing lines in target_path before reading."""
        target_path = Path(target_path)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        if temp_path.is_file():
            if target_path.is_file():
                repair_truncated_jsonl(target_path)
                with temp_path.open("r", encoding="utf-8") as tf, target_path.open("a", encoding="utf-8") as af:
                    for line in tf:
                        s = line.strip()
                        if s:
                            try:
                                json.loads(s)
                                af.write(s + "\n")
                            except Exception:
                                pass
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            else:
                try:
                    os.replace(temp_path, target_path)
                except OSError:
                    pass

        if target_path.is_file():
            repair_truncated_jsonl(target_path)

    def __init__(self, target_path: Path, append_if_exists: bool = True):
        self.target_path = Path(target_path)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.target_path.with_suffix(self.target_path.suffix + ".tmp")
        self.append = append_if_exists

        if self.append:
            self.prepare_file_for_resume(self.target_path)
            self._file = self.target_path.open("a", encoding="utf-8")
            self._is_temp = False
        else:
            self._file = self.temp_path.open("w", encoding="utf-8")
            self._is_temp = True

    def write(self, record: Dict[str, Any]) -> None:
        # allow_nan=False ensures no NaN or Infinity corrupts the JSONL
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._file.write(line + "\n")

    def flush(self) -> None:
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except (AttributeError, OSError):
            pass

    def close(self) -> None:
        if not self._file.closed:
            self.flush()
            self._file.close()

        if self._is_temp and self.temp_path.is_file():
            # Use os.replace for an atomic filesystem replacement without an empty window
            os.replace(self.temp_path, self.target_path)

    def __enter__(self) -> AtomicJsonlWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
