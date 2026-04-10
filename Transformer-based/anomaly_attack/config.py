from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    raw: Dict[str, Any]

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.raw["artifacts"]["dir"])


def load_config(path: str | Path) -> ProjectConfig:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(raw=raw)
