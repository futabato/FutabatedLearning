import unittest

import torch
from federatedlearning.server.aggregations.aggregators import average_weights


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


if __name__ == "__main__":
    unittest.main()
