"""Day-ahead backtest.

For every day d of the split, the model sees the seq_len steps that end `gap` steps before
midnight of d (actual history only) and predicts the 288 steps of d in one call.
Nothing from day d is ever an input, so this mirrors issuing a forecast before the day starts.
"""
import numpy as np
import pandas as pd
import torch

from nyload.data.dataset import DataBundle, WindowDataset
from nyload.eval.metrics import regression_metrics


@torch.no_grad()
def predict_days(model, bundle: DataBundle, split: str, device, batch_size: int = 256) -> pd.DataFrame:
    starts = bundle.window_starts(split, day_aligned=True)
    if len(starts) == 0:
        raise ValueError(f"no complete days to evaluate in split '{split}'")
    ds = WindowDataset(bundle, starts)
    model.eval()
    preds = []
    for i in range(0, len(ds), batch_size):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).to(device)
        preds.append(model(x).float().cpu().numpy())
    pred = bundle.preprocessor.inverse_target(np.concatenate(preds)).ravel()

    target_start = starts + bundle.seq_len + bundle.gap
    rows = (target_start[:, None] + np.arange(bundle.pred_len)[None, :]).ravel()
    out = bundle.frame.iloc[rows][["timestamp", "load"]].rename(columns={"load": "actual"})
    out["pred"] = pred
    return out.reset_index(drop=True)


def summarize(pred: pd.DataFrame, nyiso_forecast: pd.DataFrame | None = None) -> pd.DataFrame:
    """Metrics table. 'hourly' rows use on-the-hour points only so the NYISO forecast is comparable."""
    rows = {"model (5-min)": regression_metrics(pred["actual"], pred["pred"])}
    hourly = pred[pred["timestamp"].dt.minute == 0]
    if nyiso_forecast is not None:
        hourly = hourly.merge(nyiso_forecast, on="timestamp", how="inner")
    rows["model (hourly)"] = regression_metrics(hourly["actual"], hourly["pred"])
    if nyiso_forecast is not None and len(hourly):
        rows["NYISO forecast (hourly)"] = regression_metrics(hourly["actual"], hourly["forecast"])
    return pd.DataFrame(rows).T
