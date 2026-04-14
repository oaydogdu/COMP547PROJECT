from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)
