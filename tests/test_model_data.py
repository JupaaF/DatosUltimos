import unittest

import numpy as np
import pandas as pd
import torch

from src.data import build_dataloaders
from src.datasets import LogPowerLawDataset, MLPDataset
from src.models import LogPowerLawMLP, MLP
from src.normalization import Normalizer
from src.splitting import DataSplit


class ModelDataTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "density": [0.60, 0.61, 0.62, 0.63],
            "MCN": [4., 5., 6., 7.],
            "trace_sigma": [10., 100., 1000., 10000.],
            "q_over_p": [0.01, 0.02, 0.03, 0.04],
            "a": [0.002, 0.003, 0.004, 0.005],
            "trace_K": [0.001, 0.002, 0.003, 0.004],
            "orientation": ["horizontal", "vertical", "vertical", "vertical"],
        })
        self.validation = self.train.iloc[:2].copy()
        self.validation["MCN"] = [20., 30.]
        self.split = DataSplit(self.train, self.validation, self.validation.copy())

    def test_mlp_contract(self):
        columns = ["density", "MCN", "trace_sigma", "q_over_p", "a"]
        normalizer = Normalizer(self.train, columns)
        dataset = MLPDataset(self.train, normalizer)
        features = self.train[columns].copy()
        features["trace_sigma"] = np.log(features["trace_sigma"])
        expected = (features - features.mean()) / features.std(ddof=0)
        torch.testing.assert_close(dataset.data, torch.tensor(expected.to_numpy(), dtype=torch.float32))
        torch.testing.assert_close(dataset.targets, torch.tensor(self.train.trace_K.to_numpy(), dtype=torch.float32))
        self.assertEqual(len(dataset[0]), 2)

    def test_log_contract_and_feature_order(self):
        for columns in [["MCN"], ["a", "density"], ["trace_sigma", "MCN"]]:
            with self.subTest(columns=columns):
                normalizer = Normalizer(self.train, columns)
                dataset = LogPowerLawDataset(self.train, normalizer)
                self.assertEqual(dataset.columns, columns)
                self.assertEqual(dataset.data.shape, (4, len(columns)))
                torch.testing.assert_close(dataset.log_pressure, torch.tensor(np.log(self.train.trace_sigma.to_numpy()), dtype=torch.float32))
                torch.testing.assert_close(dataset.targets.exp(), torch.tensor(self.train.trace_K.to_numpy(), dtype=torch.float32))
                if "trace_sigma" in columns:
                    self.assertFalse(torch.allclose(dataset.data[:, 0], dataset.log_pressure))

    def test_loaders_share_training_normalizer_and_balance_only_train(self):
        loaders = build_dataloaders(self.split, LogPowerLawDataset, feature_columns=["MCN"])
        self.assertEqual(loaders.normalizer.mean["MCN"], 5.5)
        for loader in [loaders.train, loaders.validation, loaders.test]:
            self.assertIs(loader.dataset.normalizer, loaders.normalizer)
        self.assertGreater(loaders.validation.dataset.data.min().item(), 10)
        selected = list(loaders.train.sampler)
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(loaders.train.dataset.orientations[selected]), {"horizontal", "vertical"})
        self.assertEqual(list(loaders.validation.sampler), [0, 1])
        self.assertEqual(list(loaders.test.sampler), [0, 1])

    def test_both_models_can_train_on_batches(self):
        for dataset_class, model_class in [(MLPDataset, MLP), (LogPowerLawDataset, LogPowerLawMLP)]:
            loaders = build_dataloaders(self.split, dataset_class, feature_columns=["density", "MCN"], balance_orientations=False, batch_size=3)
            model = model_class(input_dim=2, dropout=0)
            optimizer = torch.optim.Adam(model.parameters())
            for batch in loaders.train:
                *inputs, target = batch
                prediction = model(*inputs)
                self.assertEqual(prediction.shape, target.shape)
                optimizer.zero_grad()
                (prediction - target).square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                optimizer.step()

    def test_logarithms_reject_invalid_physical_data(self):
        normalizer = Normalizer(self.train, ["MCN"])
        for column in ["trace_sigma", "trace_K"]:
            for value in [0., -1., float("nan"), float("inf")]:
                with self.subTest(column=column, value=value):
                    bad = self.train.copy()
                    bad.loc[0, column] = value
                    with self.assertRaises(ValueError):
                        LogPowerLawDataset(bad, normalizer)

    def test_selection_validation_and_constant_feature(self):
        for columns in [[], ["MCN", "MCN"]]:
            with self.assertRaises(ValueError):
                Normalizer(self.train, columns)
        with self.assertRaises(ValueError):
            Normalizer(self.train, ["MCN"], log_columns=["trace_sigma"])
        constant = self.train.assign(MCN=5.)
        np.testing.assert_array_equal(Normalizer(constant, ["MCN"]).transform(constant), 0)


if __name__ == "__main__":
    unittest.main()
