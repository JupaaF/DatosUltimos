"""MLP y modelo de ley de potencia con corrección neuronal en log(K)."""

import torch
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


class LogPowerLawMLP(nn.Module):
    """Predice log(K) = a + b * log(trace_sigma) + MLP(features).

    input_dim permite variar las columnas de features en futuras barridas.
    La selección y normalización de esas columnas se realiza fuera del modelo.
    log_trace_sigma debe contener el logaritmo natural de la presión física,
    sin estandarizar, con forma (batch,) o (batch, 1). features tiene forma
    (batch, input_dim). La salida tiene forma (batch,) y está en log natural:
    para recuperar K, aplicar exp; para entrenar, usar log(K) como objetivo.

    a y b son escalares entrenables junto con todos los parámetros de la MLP.
    Como la MLP puede aprender un término constante, a no tiene por sí solo
    una interpretación única; b tampoco si features permite reconstruir la
    presión. La descomposición no impone restricciones de identificabilidad.
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        a_init: float = 0.0,
        b_init: float = 0.0,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("La MLP requiere al menos una variable de entrada")
        self.input_dim = input_dim
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
        self.mlp = MLP(input_dim=input_dim, dropout=dropout)

    def forward(self, log_trace_sigma, features):
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features debe tener forma (batch, input_dim)")
        if log_trace_sigma.ndim == 2 and log_trace_sigma.shape[1] == 1:
            log_trace_sigma = log_trace_sigma.squeeze(-1)
        if log_trace_sigma.ndim != 1 or log_trace_sigma.shape[0] != features.shape[0]:
            raise ValueError("log_trace_sigma debe tener forma (batch,) o (batch, 1)")
        return self.a + self.b * log_trace_sigma + self.mlp(features)
