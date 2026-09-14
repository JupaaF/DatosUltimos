from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.train import MODEL_SPECS, TrainConfig, fit, main, run_epoch


class TrainingTests(unittest.TestCase):
    def test_epoch_mse_weights_samples_not_batches(self):
        loader = DataLoader(TensorDataset(torch.zeros(3, 1), torch.tensor([1., 1., 4.])), batch_size=2)
        model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.Flatten(0))
        self.assertAlmostEqual(run_epoch(model, loader, "cpu"), 6.)

    def test_early_stopping_restores_best_without_using_test(self):
        model = torch.nn.Linear(1, 1)
        train_loader, validation_loader = object(), object()
        loaders = SimpleNamespace(train=train_loader, validation=validation_loader)
        losses = iter([3., 1., 2., 4.])
        epoch = 0

        def fake_epoch(model, loader, device, optimizer=None):
            nonlocal epoch
            if optimizer is not None:
                self.assertIs(loader, train_loader)
                epoch += 1
                with torch.no_grad():
                    model.weight.fill_(epoch)
                return 10.
            self.assertIs(loader, validation_loader)
            return next(losses)

        with patch("src.train.run_epoch", side_effect=fake_epoch), redirect_stdout(io.StringIO()):
            history, best = fit(model, loaders, TrainConfig(epochs=10, patience=2))
        self.assertEqual(best, 2)
        self.assertEqual(len(history), 4)
        self.assertEqual(model.weight.item(), 2.)
        self.assertFalse(model.training)

    def test_cli_artifacts_reproduce_predictions_for_both_models(self):
        with TemporaryDirectory() as temp:
            for name, (model_class, _, transform) in MODEL_SPECS.items():
                output = Path(temp) / name
                with redirect_stdout(io.StringIO()):
                    main(["--model", name, "--features", "density", "MCN", "--epochs", "2",
                          "--batch-size", "32", "--output-dir", str(output)])
                checkpoint = torch.load(output / "best_model.pt", weights_only=True)
                model = model_class(**checkpoint["model_kwargs"])
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                normalizer = checkpoint["normalizer"]
                partitions = pd.read_csv(output / "partitions.csv")
                test = partitions[partitions.partition == "test"]
                features = test[normalizer["columns"]].copy()
                for column in normalizer["log_columns"]:
                    features[column] = np.log(features[column])
                features = torch.tensor(((features - normalizer["mean"]) / normalizer["scale"]).to_numpy(), dtype=torch.float32)
                inputs = [features]
                if transform == "log":
                    inputs.insert(0, torch.tensor(np.log(test.trace_sigma.to_numpy()), dtype=torch.float32))
                with torch.no_grad():
                    prediction = model(*inputs).numpy()
                saved = pd.read_csv(output / "test_predictions.csv")
                np.testing.assert_allclose(prediction, saved.prediction, rtol=1e-5, atol=1e-6)
                metrics = json.loads((output / "metrics.json").read_text())
                expected_rmse = np.sqrt(np.mean((saved.K_predicted - saved.K_true) ** 2))
                self.assertAlmostEqual(metrics["test"]["K_rmse"], expected_rmse)
                history = pd.read_csv(output / "history.csv")
                self.assertEqual(checkpoint["best_epoch"], int(history.loc[history.validation_mse.idxmin(), "epoch"]))


if __name__ == "__main__":
    unittest.main()
