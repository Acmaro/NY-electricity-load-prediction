import json
import math
import time
from pathlib import Path

import torch
from torch import nn

LOSSES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}


@torch.no_grad()
def evaluate_loss(model, loader, loss_fn, device, amp_dtype) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(x)
        total += loss_fn(pred.float(), y).item() * len(x)
        count += len(x)
    return total / max(count, 1)


def fit(model, train_loader, val_loader, cfg, device, amp_dtype, out_dir: Path, log=print) -> list[dict]:
    """Train with early stopping on validation loss. Writes best.pt, last.pt and history.json."""
    loss_fn = LOSSES[cfg.loss]()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    elif cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    else:
        scheduler = None
    grad_scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    history, best, bad_epochs = [], math.inf, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start, total, count = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if cfg.max_batches_per_epoch and i >= cfg.max_batches_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = loss_fn(model(x).float(), y)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            if cfg.grad_clip:
                grad_scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            total += loss.item() * len(x)
            count += len(x)

        val_loss = evaluate_loss(model, val_loader, loss_fn, device, amp_dtype)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": total / max(count, 1), "val_loss": val_loss,
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - start, 1)}
        history.append(row)
        log(f"epoch {epoch}/{cfg.epochs}  train {row['train_loss']:.4f}  val {val_loss:.4f}  "
            f"lr {row['lr']:.2e}  {row['seconds']}s")

        if val_loss < best:
            best, bad_epochs = val_loss, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if cfg.early_stop_patience and bad_epochs >= cfg.early_stop_patience:
                log(f"early stop: no improvement for {bad_epochs} epochs")
                break

    torch.save(model.state_dict(), out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    return history
