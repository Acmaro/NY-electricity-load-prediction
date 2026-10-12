"""nyload-eval: day-ahead backtest of a saved checkpoint.

Evaluate a training run (its data/model config and best.pt):
    nyload-eval run_dir=outputs/runs/<run> splits=[val,test]
Or any config + checkpoint, e.g. the legacy notebook weights:
    nyload-eval data=legacy_2020 model=gru model.dropout=0 checkpoint=models/legacy/GRU.pt
"""
import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from nyload import paths
from nyload.data.dataset import prepare
from nyload.experiment import evaluate_splits, setup_device
from nyload.models import build_model

log = logging.getLogger("nyload")


@hydra.main(version_base=None, config_path=str(paths.ROOT / "configs"), config_name="eval")
def main(cfg: DictConfig) -> None:
    paths.load_dotenv()
    if cfg.get("run_dir"):
        run_dir = paths.resolve(cfg.run_dir)
        saved = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
        cfg.data, cfg.model = saved.data, saved.model
        cfg.checkpoint = cfg.checkpoint or str(run_dir / "best.pt")
    if not cfg.get("checkpoint"):
        raise ValueError("pass run_dir=<training run> or checkpoint=<path to .pt>")
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    device, _ = setup_device(cfg.env, log.info)

    bundle = prepare(cfg.data)
    model = build_model(cfg.model, bundle.num_features, cfg.data.pred_len).to(device)
    state = torch.load(paths.resolve(cfg.checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)
    log.info(f"loaded {cfg.checkpoint}")
    evaluate_splits(model, bundle, cfg, device, out_dir, splits=tuple(cfg.splits), log=log.info)
    log.info(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
