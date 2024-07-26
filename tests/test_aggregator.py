import unittest

import torch
from federatedlearning.server.aggregations.aggregators import (
    average_weights,
    krum,
    median_weights,
)


class TestAverageWeights(unittest.TestCase):
    def test_average_of_empty_list(self) -> None:
        with self.assertRaises(IndexError):
            average_weights([])  # Empty list should raise an IndexError

    def test_average_with_single_weight(self) -> None:
        weight: list[dict[str, torch.Tensor]] = [
            {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        ]
        expected_result: dict[str, torch.Tensor] = {
            "a": torch.tensor(1.0),
            "b": torch.tensor(2.0),
        }
        result: dict[str, torch.Tensor] = average_weights(weight)
        for key in expected_result.keys():
            self.assertTrue(
                torch.equal(expected_result[key], result[key]),
                "Average with single weight should return the weight itself.",
            )

    def test_average_weights(self) -> None:
        weights: list[dict[str, torch.Tensor]] = [
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
            {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([7.0, 8.0])},
            {"a": torch.tensor([9.0, 10.0]), "b": torch.tensor([11.0, 12.0])},
        ]
        expected_result: dict[str, torch.Tensor] = {
            "a": torch.tensor([5.0, 6.0]),  # (1+5+9)/3, (2+6+10)/3
            "b": torch.tensor([7.0, 8.0]),  # (3+7+11)/3, (4+8+12)/3
        }
        result: dict[str, torch.Tensor] = average_weights(weights)
        for key in expected_result.keys():
            self.assertTrue(
                torch.allclose(expected_result[key], result[key]),
                f"Weights averaged incorrectly for key '{key}'.",
            )


class TestMedianWeights(unittest.TestCase):
    def test_median_of_empty_list(self) -> None:
        with self.assertRaises(IndexError):
            median_weights([])  # Empty list should raise an IndexError

    def test_median_with_single_weight(self) -> None:
        weight: list[dict[str, torch.Tensor]] = [
            {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
        ]
        expected_result: dict[str, torch.Tensor] = {
            "a": torch.tensor(1.0),
            "b": torch.tensor(2.0),
        }
        result: dict[str, torch.Tensor] = median_weights(weight)
        for key in expected_result.keys():
            self.assertTrue(
                torch.equal(expected_result[key], result[key]),
                "Median with single weight should return the weight itself.",
            )

    def test_median_weights(self) -> None:
        weights: list[dict[str, torch.Tensor]] = [
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])},
            {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([7.0, 8.0])},
            {"a": torch.tensor([9.0, 10.0]), "b": torch.tensor([11.0, 12.0])},
        ]
        expected_result: dict[str, torch.Tensor] = {
            "a": torch.tensor(
                [5.0, 6.0]
            ),  # Median of [1, 5, 9] and [2, 6, 10]
            "b": torch.tensor(
                [7.0, 8.0]
            ),  # Median of [3, 7, 11] and [4, 8, 12]
        }
        result: dict[str, torch.Tensor] = median_weights(weights)
        for key in expected_result.keys():
            self.assertTrue(
                torch.equal(expected_result[key], result[key]),
                f"Weights median incorrectly for key '{key}'.",
            )


class TestKrumAlgorithm(unittest.TestCase):
    def test_krum_basic(self) -> None:
        weights = [
            {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 4.0]),
            },  # Benign
            {
                "a": torch.tensor([1.1, 2.1]),
                "b": torch.tensor([3.1, 4.1]),
            },  # Benign
            {
                "a": torch.tensor([10.0, 20.0]),
                "b": torch.tensor([30.0, 40.0]),
            },  # Malicious
        ]
        f = 1
        # Weights to be considered most normal.
        expected_result: dict[str, torch.Tensor] = {
            "a": torch.tensor([1.0, 2.0]),
            "b": torch.tensor([3.0, 4.0]),
        }
        result: dict[str, torch.Tensor] = krum(weights, f)
        for key in expected_result.keys():
            self.assertTrue(
                torch.equal(expected_result[key], result[key]),
                f"Weights median incorrectly for key '{key}'.",
            )

    def test_not_enough_weights(self) -> None:
        weights = [
            {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
        ]
        f = 1

        with self.assertRaises(ValueError):
            krum(weights, f)

    def test_all_same_weights(self) -> None:
        weights = [
            {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 1.0])},
            {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 1.0])},
            {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 1.0])},
        ]
        f = 1
        # All weights are equal, so any of them can be chosen
        expected_result: dict[str, torch.Tensor] = {
            "a": torch.tensor([1.0, 1.0]),
            "b": torch.tensor([1.0, 1.0]),
        }
        result: dict[str, torch.Tensor] = krum(weights, f)
        for key in expected_result.keys():
            self.assertTrue(
                torch.equal(expected_result[key], result[key]),
                f"Weights median incorrectly for key '{key}'.",
            )


if __name__ == "__main__":
    unittest.main()
