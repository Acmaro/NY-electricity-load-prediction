"""Feature selection and standardization. Statistics are fit on the training split only."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    features: list[str]          # model input columns; features[0] must be the target
    scale: list[str]             # columns to standardize
    target: str = "load"
    stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.features[0] != self.target:
            raise ValueError(f"the first feature must be the target '{self.target}', got {self.features}")
        unknown = set(self.scale) - set(self.features)
        if unknown:
            raise ValueError(f"scale columns not in features: {sorted(unknown)}")

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.stats = {c: (float(df[c].mean()), float(df[c].std(ddof=0)) or 1.0) for c in self.scale}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = df[self.features].astype("float64").copy()
        for c, (mean, std) in self.stats.items():
            out[c] = (out[c] - mean) / std
        return out.to_numpy(dtype=np.float32)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        if self.target not in self.stats:
            return np.asarray(values)
        mean, std = self.stats[self.target]
        return np.asarray(values) * std + mean

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        d = json.loads(Path(path).read_text())
        d["stats"] = {k: tuple(v) for k, v in d["stats"].items()}
        return cls(**d)
