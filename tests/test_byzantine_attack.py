import torch

from attack.byzantines import no_byzantine, labelflip_attack


def test_no_byzantine():
    expected = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    f = 0

    # Byzantine Attack
    actual = no_byzantine(expected, f)

    # Verify result
    assert actual == expected


def test_labelflip_attack():
    tensor = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], device="cpu")
    expected = torch.Tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9], device="cpu")

    # Byzantine Attack
    actual = labelflip_attack(tensor)

    # Verify result
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    test_no_byzantine()
    test_labelflip_attack()
