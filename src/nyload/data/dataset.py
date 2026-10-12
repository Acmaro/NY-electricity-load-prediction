"""Load processed years, build splits and sliding-window datasets.

All splits share one frame. A window is (history of seq_len steps, next pred_len target steps,
optionally `gap` steps later). A window belongs to a split when its *target* lies inside that
split's period; history may reach back into earlier data, which is information available at
forecast time. Windows touching source outages (load_gap) or a hole between loaded years are dropped.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nyload.data.build import FREQ, STEPS_PER_DAY, forecast_path, processed_path
from nyload.features.calendar import add_calendar_features
from nyload.features.preprocess import Preprocessor

STEP_NS = pd.Timedelta(FREQ).value


def load_years(zone: str, years: list[int]) -> pd.DataFrame:
    missing = [y for y in years if not processed_path(zone, y).exists()]
    if missing:
        raise FileNotFoundError(f"processed data missing for {missing}; run `nyload-data all --years ...`")
    df = pd.concat([pd.read_parquet(processed_path(zone, y)) for y in years], ignore_index=True)
    return df.sort_values("timestamp", ignore_index=True)


def load_forecast(zone: str, years: list[int]) -> pd.DataFrame | None:
    frames = [pd.read_parquet(p) for y in years if (p := forecast_path(zone, y)).exists()]
    return pd.concat(frames, ignore_index=True) if frames else None


@dataclass
class DataBundle:
    frame: pd.DataFrame            # timestamp, raw columns, calendar features
    values: np.ndarray             # (n, num_features) model inputs, float32, column 0 = scaled target
    masks: dict[str, np.ndarray]   # split name -> boolean row mask over `frame`
    preprocessor: Preprocessor
    seq_len: int
    pred_len: int
    gap: int

    @property
    def num_features(self) -> int:
        return self.values.shape[1]

    def window_starts(self, split: str, stride: int = 1, day_aligned: bool = False) -> np.ndarray:
        n = len(self.frame)
        span = self.seq_len + self.gap + self.pred_len
        starts = np.arange(0, n - span + 1)
        target_start = starts + self.seq_len + self.gap
        target_end = target_start + self.pred_len  # exclusive

        in_split = np.concatenate([[0], np.cumsum(self.masks[split])])
        ok = (in_split[target_end] - in_split[target_start]) == self.pred_len

        bad = np.concatenate([[0], np.cumsum(self.frame["load_gap"].to_numpy() > 0)])
        ok &= (bad[starts + span] - bad[starts]) == 0

        t = self.frame["timestamp"].to_numpy().astype("int64")
        ok &= (t[starts + span - 1] - t[starts]) == (span - 1) * STEP_NS  # no hole between years

        if day_aligned:
            ts = self.frame["timestamp"]
            ok &= ((ts.dt.hour == 0) & (ts.dt.minute == 0)).to_numpy()[target_start]
        starts = starts[ok]
        return starts[::stride] if stride > 1 else starts


def prepare(cfg) -> DataBundle:
    """Build the shared frame, fit the preprocessor on the training period and make split masks."""
    train_years = list(cfg.train_years)
    val_years = list(cfg.get("val_years") or [])
    test_years = list(cfg.test_years)
    # one extra year before each evaluated year so its first days have history
    needed = set(train_years) | set(val_years) | set(test_years)
    needed |= {y - 1 for y in val_years + test_years}
    available = [y for y in sorted(needed) if processed_path(cfg.zone, y).exists()]
    if not set(train_years + val_years + test_years) <= set(available):
        load_years(cfg.zone, train_years + val_years + test_years)  # raises with the missing list
    frame = load_years(cfg.zone, available)
    frame = add_calendar_features(frame, list(cfg.features), cfg.holidays.country, cfg.holidays.subdiv)

    year = frame["timestamp"].dt.year
    masks = {"train": year.isin(train_years).to_numpy(), "test": year.isin(test_years).to_numpy()}
    if val_years:
        masks["val"] = year.isin(val_years).to_numpy()
    else:
        # carve the last val_fraction of the training period, cut at a day boundary
        idx = np.flatnonzero(masks["train"])
        cut = idx[int(len(idx) * (1 - cfg.val_fraction)) // STEPS_PER_DAY * STEPS_PER_DAY]
        masks["val"] = masks["train"] & (np.arange(len(frame)) >= cut)
        masks["train"] = masks["train"] & (np.arange(len(frame)) < cut)

    pre = Preprocessor(features=list(cfg.features), scale=list(cfg.scale))
    fit_rows = frame[masks["train"] & (frame["load_gap"].to_numpy() == 0)]
    pre.fit(fit_rows)
    return DataBundle(frame=frame, values=pre.transform(frame), masks=masks, preprocessor=pre,
                      seq_len=cfg.seq_len, pred_len=cfg.pred_len, gap=cfg.get("gap", 0))


class WindowDataset(Dataset):
    def __init__(self, bundle: DataBundle, starts: np.ndarray):
        self.values = torch.from_numpy(bundle.values)
        self.starts = torch.as_tensor(starts, dtype=torch.long)
        self.seq_len, self.pred_len, self.gap = bundle.seq_len, bundle.pred_len, bundle.gap

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = int(self.starts[i])
        x = self.values[s:s + self.seq_len]
        t = s + self.seq_len + self.gap
        y = self.values[t:t + self.pred_len, 0]
        return x, y
