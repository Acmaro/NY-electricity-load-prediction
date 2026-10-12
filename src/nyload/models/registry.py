"""Model registry. A model maps history (batch, seq_len, num_features) -> (batch, pred_len).

Add a model:
    @register("tcn")
    class TCNForecaster(nn.Module):
        def __init__(self, num_features, output_size, **hparams): ...
plus configs/model/tcn.yaml with `name: tcn` and the hparams.
"""
MODELS: dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        if name in MODELS:
            raise KeyError(f"model '{name}' registered twice")
        MODELS[name] = cls
        return cls
    return wrap


def build_model(model_cfg, num_features: int, output_size: int):
    params = dict(model_cfg)
    name = params.pop("name")
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}', available: {sorted(MODELS)}")
    return MODELS[name](num_features=num_features, output_size=output_size, **params)
