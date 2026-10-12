import torch
from torch import nn

from nyload.models.registry import register


class _RNNForecaster(nn.Module):
    """Stacked RNN; the last hidden state is mapped to all pred_len outputs at once."""
    cell: type[nn.Module]
    attr: str  # submodule name, kept as in the original notebook so legacy checkpoints load

    def __init__(self, num_features: int, output_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        rnn = self.cell(num_features, hidden_size, num_layers, batch_first=True,
                        dropout=dropout if num_layers > 1 else 0.0)
        setattr(self, self.attr, rnn)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = getattr(self, self.attr)(x)
        return self.fc(out[:, -1, :])


@register("lstm")
class LSTMForecaster(_RNNForecaster):
    cell, attr = nn.LSTM, "lstm"


@register("gru")
class GRUForecaster(_RNNForecaster):
    cell, attr = nn.GRU, "gru"
