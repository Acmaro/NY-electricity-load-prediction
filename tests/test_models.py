import pytest
import torch
from omegaconf import OmegaConf

from nyload import paths
from nyload.models import MODELS, build_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_output_shape_and_one_batch_overfit(name):
    torch.manual_seed(0)
    model = build_model(OmegaConf.create({"name": name, "hidden_size": 16, "num_layers": 1}), 5, 288)
    x, y = torch.randn(8, 48, 5), torch.randn(8, 288)
    assert model(x).shape == (8, 288)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        loss = torch.nn.functional.mse_loss(model(x), y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


@pytest.mark.parametrize("name,file", [("gru", "GRU.pt"), ("lstm", "LSTM.pt")])
def test_legacy_checkpoints_load(name, file):
    path = paths.ROOT / "models" / "legacy" / file
    model = build_model(OmegaConf.create({"name": name, "hidden_size": 64, "num_layers": 2}), 7, 288)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
