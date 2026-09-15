"""Modelos neuronales y RBF para predecir la conductividad."""

import math

import numpy as np
from scipy.interpolate import RBFInterpolator
import torch
from torch import nn
from torch.nn import functional as F


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


class ConditionalLogPowerLawMLP(nn.Module):
    """Ley de potencia cuyo intercepto y exponente dependen de las features.

    Predice ``target = A(x) + softplus(B(x)) * standardized_log_pressure``.
    La presión se centra y escala con estadísticas exclusivas de train. El
    ``softplus`` impone una pendiente positiva respecto a ``ln(p)``.
    """

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        pressure_mean: float = 0.0,
        pressure_scale: float = 1.0,
        hidden_sizes: tuple[int, ...] | list[int] = (32, 16),
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("La red requiere al menos una variable de entrada")
        if not hidden_sizes or any(size < 1 for size in hidden_sizes):
            raise ValueError("hidden_sizes debe contener tamaños positivos")
        if not math.isfinite(pressure_mean) or not math.isfinite(pressure_scale) or pressure_scale <= 0:
            raise ValueError("La normalización de presión debe ser finita y positiva")
        self.input_dim = input_dim
        self.register_buffer("pressure_mean", torch.tensor(float(pressure_mean)))
        self.register_buffer("pressure_scale", torch.tensor(float(pressure_scale)))
        sizes = [input_dim, *hidden_sizes]
        layers = []
        for source, destination in zip(sizes, sizes[1:]):
            layers.extend((nn.Linear(source, destination), nn.Tanh(), nn.Dropout(dropout)))
        layers.append(nn.Linear(sizes[-1], 2))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [layer for layer in self.network if isinstance(layer, nn.Linear)]
        for index, layer in enumerate(layers):
            gain = nn.init.calculate_gain("tanh") if index < len(layers) - 1 else 1.0
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)
        # softplus(inverse_softplus(1)) = 1: pendiente inicial bien escalada.
        layers[-1].bias.data[1] = math.log(math.expm1(1.0))

    def forward(self, log_trace_sigma, features):
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("features debe tener forma (batch, input_dim)")
        if log_trace_sigma.ndim == 2 and log_trace_sigma.shape[1] == 1:
            log_trace_sigma = log_trace_sigma.squeeze(-1)
        if log_trace_sigma.ndim != 1 or log_trace_sigma.shape[0] != features.shape[0]:
            raise ValueError("log_trace_sigma debe tener forma (batch,) o (batch, 1)")
        parameters = self.network(features)
        intercept = parameters[:, 0]
        slope = F.softplus(parameters[:, 1])
        normalized_pressure = (log_trace_sigma - self.pressure_mean) / self.pressure_scale
        return intercept + slope * normalized_pressure


class RBFModel:
    """Envoltorio de SciPy para las arquitecturas RBF del estudio externo.

    Recibe matrices NumPy ``(n_samples, n_features)`` y devuelve un vector.
    La transformación opcional reproduce el min-max a ``[-1, 1]`` de
    ``rbf_architecture_study.py``. No transforma el objetivo: el llamador debe
    proporcionar ``log10(K)`` para reproducir el entrenamiento del supervisor.
    """

    name = "rbf"
    kernel: str
    degree: int
    smoothing: float
    neighbors: int | None

    def __init__(self, *, scale_features: bool = False):
        self.scale_features = scale_features
        self.x_min_: np.ndarray | None = None
        self.x_scale_: np.ndarray | None = None
        self.effective_neighbors_: int | None = None
        self.model: RBFInterpolator | None = None

    @staticmethod
    def _validate_features(X, *, expected_features: int | None = None) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or len(values) == 0 or values.shape[1] == 0:
            raise ValueError("X debe tener forma (n_samples, n_features) y no estar vacío")
        if expected_features is not None and values.shape[1] != expected_features:
            raise ValueError(f"X debe tener {expected_features} variables de entrada")
        if not np.isfinite(values).all():
            raise ValueError("X requiere valores finitos")
        return values

    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        if not self.scale_features:
            return X
        self.x_min_ = X.min(axis=0)
        x_max = X.max(axis=0)
        self.x_scale_ = np.where((x_max - self.x_min_) < 1e-12, 1.0, x_max - self.x_min_)
        return 2.0 * (X - self.x_min_) / self.x_scale_ - 1.0

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if not self.scale_features:
            return X
        return 2.0 * (X - self.x_min_) / self.x_scale_ - 1.0

    def fit(self, X, y):
        X = self._validate_features(X)
        target = np.asarray(y, dtype=float)
        if target.ndim not in (1, 2) or target.size != len(X):
            raise ValueError("y debe contener un objetivo por muestra")
        target = target.reshape(-1)
        if not np.isfinite(target).all():
            raise ValueError("y requiere valores finitos")
        neighbors = self.neighbors
        if neighbors is not None:
            polynomial_terms = math.comb(X.shape[1] + self.degree, self.degree)
            if len(X) < polynomial_terms:
                raise ValueError(
                    f"Se requieren al menos {polynomial_terms} muestras para "
                    f"{X.shape[1]} entradas y degree={self.degree}"
                )
            neighbors = min(max(neighbors, polynomial_terms), len(X))
        self.effective_neighbors_ = neighbors
        self.model = RBFInterpolator(
            self._fit_transform_features(X),
            target.reshape(-1, 1),
            kernel=self.kernel,
            degree=self.degree,
            smoothing=self.smoothing,
            neighbors=neighbors,
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("El modelo debe ajustarse antes de predecir")
        expected_features = self.model.y.shape[1]
        X = self._validate_features(X, expected_features=expected_features)
        return self.model(self._transform_features(X)).ravel()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kernel": self.kernel,
            "degree": self.degree,
            "smoothing": self.smoothing,
            "neighbors": self.neighbors,
            "effective_neighbors": self.effective_neighbors_,
            "scale_features": self.scale_features,
            "x_min": None if self.x_min_ is None else self.x_min_.tolist(),
            "x_scale": None if self.x_scale_ is None else self.x_scale_.tolist(),
        }


class RBF4(RBFModel):
    """RBF cúbica flexible/local: configuración ``rbf_4``."""

    name = "rbf_4"
    kernel = "cubic"
    degree = 1
    smoothing = 1e-8
    neighbors = 40


class RBF5(RBFModel):
    """RBF quíntica más local: configuración ``rbf_5``."""

    name = "rbf_5"
    kernel = "quintic"
    degree = 2
    smoothing = 1e-10
    neighbors = 25


RBF_MODELS = {"rbf_4": RBF4, "rbf_5": RBF5}
