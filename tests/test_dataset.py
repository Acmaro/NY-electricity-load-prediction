import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from nyload.data.build import processed_path
from nyload.data.dataset import prepare
from nyload.eval.backtest import predict_days

CFG = OmegaConf.create({
    "zone": "CAPITL", "train_years": [2020], "val_years": [], "val_fraction": 0.2, "test_years": [2021],
    "seq_len": 288, "pred_len": 288, "gap": 0, "train_stride": 1, "val_stride": 12,
    "features": ["load", "temp", "is_workday", "hour_sin", "hour_cos"], "scale": ["load", "temp"],
    "holidays": {"country": "US", "subdiv": None},
})


def test_targets_stay_inside_their_split(data_dir):
    b = prepare(CFG)
    ts = b.frame["timestamp"]
    for split in ("train", "val", "test"):
        starts = b.window_starts(split)
        first_t, last_t = starts + b.seq_len, starts + b.seq_len + b.pred_len - 1
        assert b.masks[split][first_t].all() and b.masks[split][last_t].all()
    # validation is the tail of 2020, test starts on 2021-01-01 with 2020-12-31 as history
    assert ts[b.window_starts("val")[0] + b.seq_len] > pd.Timestamp("2020-10-01")
    test_days = b.window_starts("test", day_aligned=True)
    assert ts[test_days[0] + b.seq_len] == pd.Timestamp("2021-01-01")
    assert len(test_days) == 365


def test_scaler_is_fit_on_training_rows_only(data_dir):
    b = prepare(CFG)
    train_load = b.frame.loc[b.masks["train"], "load"]
    assert b.preprocessor.stats["load"][0] == pytest.approx(train_load.mean())


def test_windows_touching_an_outage_are_dropped(data_dir):
    df = pd.read_parquet(processed_path("CAPITL", 2021))
    df.loc[1000:1100, "load_gap"] = 1
    df.to_parquet(processed_path("CAPITL", 2021), index=False)
    b = prepare(CFG)
    starts = b.window_starts("test")
    offset = int(np.flatnonzero(b.masks["test"])[0])
    span = b.seq_len + b.pred_len
    bad = np.arange(offset + 1000, offset + 1101)
    assert not any(((s <= bad) & (bad < s + span)).any() for s in starts)


class _MeanOfHistory(torch.nn.Module):
    def forward(self, x):
        return x[:, :, 0].mean(dim=1, keepdim=True).repeat(1, 288)


def test_backtest_never_uses_the_target_day(data_dir):
    b = prepare(CFG)
    before = predict_days(_MeanOfHistory(), b, "test", torch.device("cpu"))
    day = b.frame["timestamp"].dt.date == pd.Timestamp("2021-03-10").date()
    b.values[day.to_numpy(), 0] += 100.0  # tamper with the actual load of one target day
    after = predict_days(_MeanOfHistory(), b, "test", torch.device("cpu"))
    on_day = before["timestamp"].dt.date == pd.Timestamp("2021-03-10").date()
    assert np.allclose(before.loc[on_day, "pred"], after.loc[on_day, "pred"])
    assert not np.allclose(before["pred"], after["pred"])  # later days do see it as history
