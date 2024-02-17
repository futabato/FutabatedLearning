import torch
from attack.byzantines import (
    bitflip_attack,
    chosen_labelflip_attack,
    labelflip_attack,
    no_byzantine,
)


def test_no_byzantine() -> None:
    tensor: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    expected: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")

    # Byzantine Attack
    actual: torch.Tensor = no_byzantine(tensor)

    # Verify result
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_labelflip_attack() -> None:
    tensor: torch.Tensor = torch.Tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0], device="cpu"
    )
    expected: torch.Tensor = torch.Tensor(
        [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 9.0], device="cpu"
    )

    # Byzantine Attack
    actual: torch.Tensor = labelflip_attack(tensor)

    # Verify result
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_bitflip_attack() -> None:
    tensor: torch.Tensor = torch.Tensor(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.2, 0.4, 0.6, 0.8],
                [0.0, 0.5, 0.5, 0.0],
                [0.1, 0.3, 0.5, 0.7],
            ]
        ],
        device="cpu",
    )
    expected: torch.Tensor = torch.Tensor(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.8, 0.6, 0.4, 0.2],
                [1.0, 0.5, 0.5, 1.0],
                [0.9, 0.7, 0.5, 0.3],
            ],
        ],
        device="cpu",
    )
    # Byzantine Attack
    actual: torch.Tensor = bitflip_attack(tensor)

    # Verify result
    assert len(actual) == len(expected)
    torch.testing.assert_close(actual, expected)


def test_chosen_labelflip_attack() -> None:
    tensor: torch.Tensor
    expected: torch.Tensor
    actual: torch.Tensor
    for src_label in range(10):
        for dst_label in range(10):
            tensor = torch.Tensor(
                [
                    src_label,
                    dst_label,
                ],
                device="cpu",
            )
            expected = torch.Tensor(
                [
                    dst_label,
                    dst_label,
                ],
                device="cpu",
            )

            # Byzantine Attack
            actual = chosen_labelflip_attack(tensor, src_label, dst_label)

            # Verify result
            assert actual.shape == expected.shape
            assert torch.equal(actual, expected)


if __name__ == "__main__":
    test_no_byzantine()
    test_labelflip_attack()
    test_bitflip_attack()
    test_chosen_labelflip_attack()
