import torch

from attack.byzantines import no_byzantine


def test_no_byzantine():
    expected = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    f = 0

    # Byzantine Attack
    actual = no_byzantine(expected, f)

    # Verify result
    assert actual == expected


if __name__ == "__main__":
    test_no_byzantine()
