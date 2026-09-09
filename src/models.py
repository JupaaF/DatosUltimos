"""MLP de tres capas con pesos: dos ocultas y una salida lineal."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier para las capas tanh y la salida lineal; sesgos a cero."""
        layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        for index, layer in enumerate(layers):
            gain = nn.init.calculate_gain("tanh") if index < len(layers) - 1 else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)
