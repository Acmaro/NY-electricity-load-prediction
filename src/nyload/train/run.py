"""nyload-train: train a model from the Hydra config, then backtest it.

    nyload-train model=gru env=local_win
    nyload-train data=recent_2019_2021 model=lstm train.epochs=50 model.hidden_size=128
    nyload-train train=fast_debug env=cpu
"""
import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from nyload import paths
from nyload.data.dataset import prepare
from nyload.experiment import evaluate_splits, make_loader, setup_device
from nyload.models import build_model
from nyload.train.trainer import fit
from nyload.utils.device import seed_everything

log = logging.getLogger("nyload")


@hydra.main(version_base=None, config_path=str(paths.ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> float:
    paths.load_dotenv()
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    log.info("\n" + OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    device, amp_dtype = setup_device(cfg.env, log.info)

    bundle = prepare(cfg.data)
    bundle.preprocessor.save(out_dir / "preprocessor.json")
    train_loader = make_loader(bundle, "train", cfg.train.batch_size, cfg.env, device,
                               shuffle=True, stride=cfg.data.train_stride)
    val_loader = make_loader(bundle, "val", cfg.train.batch_size * 4, cfg.env, device,
                             shuffle=False, stride=cfg.data.val_stride)
    log.info(f"windows: train {len(train_loader.dataset)}, val {len(val_loader.dataset)}, "
             f"features {bundle.preprocessor.features}")

    model = build_model(cfg.model, bundle.num_features, cfg.data.pred_len).to(device)
    log.info(f"{cfg.model.name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    train_model = torch.compile(model) if cfg.env.compile else model

    history = fit(train_model, train_loader, val_loader, cfg.train, device, amp_dtype, out_dir, log.info)
    evaluate_splits(model, bundle, cfg, device, out_dir, log=log.info)
    log.info(f"outputs: {out_dir}")
    return min(h["val_loss"] for h in history)  # objective for Hydra sweepers


if __name__ == "__main__":
    main()
