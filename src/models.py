"""Candidatos para predecir trace_K; todos usan las mismas entradas."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model_candidates(seed=42):
    """La estandarización de X e y se ajusta exclusivamente durante fit."""
    regressors = {
        # Tres capas con pesos: 5 -> 32 -> 16 -> 1 (dos ocultas y salida).
        "mlp_3_layers": MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="tanh", solver="lbfgs",
            alpha=1e-3, max_iter=5000, max_fun=50000, tol=1e-7,
            random_state=seed,
        ),
        **{f"ridge_alpha_{alpha}": Ridge(alpha=alpha) for alpha in (0.01, 0.1, 1, 10, 100)},
        "gaussian_process": GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
            + WhiteKernel(1e-3, (1e-8, 1)),
            random_state=seed, n_restarts_optimizer=2,
        ),
        **{f"extra_trees_leaf_{leaf}": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=leaf, random_state=seed, n_jobs=-1,
        ) for leaf in (1, 3, 5)},
    }
    for name, regressor in regressors.items():
        for target in ("direct", "log"):
            model = TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), regressor),
                transformer=StandardScaler(),
            )
            if target == "log":
                model = TransformedTargetRegressor(
                    regressor=model, func=np.log, inverse_func=np.exp,
                )
            yield name, target, model
