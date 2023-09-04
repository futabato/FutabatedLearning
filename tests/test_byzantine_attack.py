import torch

from attack.byzantines import bitflip_attack, labelflip_attack, no_byzantine


def test_no_byzantine():
    tensor = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    expected = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    f = 0

    # Byzantine Attack
    actual = no_byzantine(tensor, f)

    # Verify result
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_labelflip_attack():
    tensor = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0], device="cpu")
    expected = torch.Tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9], device="cpu")

    # Byzantine Attack
    actual = labelflip_attack(tensor)

    # Verify result
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_bitflip_attack():
    v = [
        torch.tensor([1, 1, 0, 0]),
        torch.tensor([1, 0, 1, 0]),
        torch.tensor([1, 0, 0, 1]),
        torch.tensor([0, 1, 1, 0]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0, 0, 1, 1]),
    ]
    expected = [
        # num_byzantine = 0
        [
            torch.tensor([1, 1, 0, 0]),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        # num_byzantine = 1
        [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        # num_byzantine = 2
        [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        # num_byzantine = 3
        [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        # num_byzantine = 4
        [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 0, 1, 1]),
        ],
        # num_byzantine = 5
        [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([0, 0, 1, 1]),
        ],
        # num_byzantine = 6
        [
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 1, 0]),
            torch.tensor([1, 0, 0, 1]),
            torch.tensor([1, 0, 1, 0]),
            torch.tensor([1, 1, 0, 0]),
        ],
    ]
    for f in range(len(v) + 1):
        actual = bitflip_attack(v, f)
        assert len(actual) == len(expected[f])
        assert torch.equal(actual[0], expected[f][0])


if __name__ == "__main__":
    test_no_byzantine()
    test_labelflip_attack()
    test_bitflip_attack()
