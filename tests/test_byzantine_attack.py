import torch
from attack.byzantines import (
    bitflip_attack,
    chosen_labelflip_attack,
    gaussian_attack,
    labelflip_attack,
)


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


def test_gaussian_attack() -> None:
    device: torch.device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # Create a fake weight dictionary with one tensor
    fake_weight: dict[str, torch.Tensor] = {
        "layer_1.weight": torch.zeros(3, 3, device=device),
    }

    # Call the function under test
    attacked_weights: dict[str, torch.Tensor] = gaussian_attack(
        fake_weight, device
    )

    # Check that the shapes match
    for name, tensor in fake_weight.items():
        assert (
            attacked_weights[name].shape == tensor.shape
        ), f"Shape mismatch for {name}"

    # Optionally check the properties of the Gaussian noise (mean and std deviation)
    # Here we use a large number of samples to approximate the mean and std dev
    large_fake_weight: dict[str, torch.Tensor] = {
        "large_layer.weight": torch.zeros(1000, 1000, device=device)
    }
    large_attacked_weights: dict[str, torch.Tensor] = gaussian_attack(
        large_fake_weight, device
    )
    noise_added = large_attacked_weights["large_layer.weight"].flatten()
    mean_of_noise = noise_added.mean().item()
    std_of_noise = noise_added.std().item()

    # Assert that the mean and std dev are close to their expected values
    assert (
        abs(mean_of_noise) < 1e-2
    ), f"Mean of noise is too far from 0.0: {mean_of_noise}"
    assert (
        abs(std_of_noise - 0.01) < 1e-2
    ), f"Std deviation of noise is too far from 0.01: {std_of_noise}"


def test_bitflip_attack() -> None:
    # Create a fake weight dictionary with one tensor
    fake_weight: dict[str, torch.Tensor] = {
        "layer_1.weight": torch.tensor([[0.5, -0.2], [0.3, -0.4]]),
        "layer_2.bias": torch.tensor([0.1, -0.1]),
    }

    # Expected results after bitflip attack
    expected_flipped_weights: dict[str, torch.Tensor] = {
        "layer_1.weight": torch.tensor([[0.5, 1.2], [0.7, 1.4]]),
        "layer_2.bias": torch.tensor([0.9, 1.1]),
    }

    # Call the function under test
    flipped_weights: dict[str, torch.Tensor] = bitflip_attack(fake_weight)

    # Check that the returned dictionary has the same keys and corresponding shapes
    assert set(flipped_weights.keys()) == set(
        fake_weight.keys()
    ), "Keys of the dictionaries do not match."
    for name in fake_weight:
        assert (
            flipped_weights[name].shape == expected_flipped_weights[name].shape
        ), f"Shape mismatch for {name}"

    # Check that the bit flip is correctly applied
    for name, tensor in expected_flipped_weights.items():
        flipped_tensor = flipped_weights[name]
        assert torch.allclose(
            flipped_tensor, tensor
        ), f"Bitflip operation failed for {name}"


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
    test_labelflip_attack()
    test_gaussian_attack()
    test_bitflip_attack()
    test_chosen_labelflip_attack()
