"""Pieces shared by the train and eval entry points."""
import json
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nyload.data.dataset import DataBundle, WindowDataset, load_forecast
from nyload.eval.backtest import predict_days, summarize
from nyload.utils.device import resolve_device, resolve_precision


def setup_device(env: DictConfig, log=print):
    device = resolve_device(env.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log(f"device: cuda ({torch.cuda.get_device_name(device)})")
    else:
        log(f"device: {device}")
    return device, resolve_precision(env.precision, device)


def make_loader(bundle: DataBundle, split: str, batch_size: int, env: DictConfig, device,
                shuffle: bool, stride: int = 1) -> DataLoader:
    ds = WindowDataset(bundle, bundle.window_starts(split, stride=stride))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=env.num_workers,
                      pin_memory=device.type == "cuda", persistent_workers=env.num_workers > 0,
                      drop_last=False)


def evaluate_splits(model, bundle: DataBundle, cfg: DictConfig, device, out_dir: Path,
                    splits=("val", "test"), log=print) -> dict:
    results = {}
    for split in splits:
        pred = predict_days(model, bundle, split, device)
        forecast = load_forecast(cfg.data.zone, sorted(pred["timestamp"].dt.year.unique().tolist()))
        pred.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
        table = summarize(pred, forecast)
        results[split] = table.to_dict(orient="index")
        log(f"\n[{split}] {pred['timestamp'].dt.date.nunique()} days\n"
            + table.to_string(float_format=lambda v: f"{v:.3f}"))
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
