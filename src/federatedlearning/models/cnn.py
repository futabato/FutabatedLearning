import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class CNNMnist(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        """
        A Convolutional Neural Network designed for the MNIST dataset.

        Args:
            cfg (DictConfig): A configuration object specifying hyperparameters.
        """
        super(CNNMnist, self).__init__()
        # First convolutional layer taking in 'num_channels' from config and outputting 10 channels
        self.conv1 = nn.Conv2d(cfg.train.num_channels, 10, kernel_size=5)
        # Second convolutional layer taking in 10 channels and outputting 20 channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer for second convolutional layer to prevent overfitting
        self.conv2_drop = nn.Dropout2d()
        # First fully connected layer with flattened input and 50 output units
        self.fc1 = nn.Linear(320, 50)
        # Second fully connected layer that outputs the number of classes from config
        self.fc2 = nn.Linear(50, cfg.train.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network with ReLU activations and pooling layers.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The output tensor after applying log softmax.
        """
        # Apply convolutional layers, ReLU activations, max pooling, and dropout
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor for feeding into fully connected layers
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout to the output of the first fully connected layer during training
        x = F.dropout(x, training=self.training)
        # Apply second fully connected layer
        x = self.fc2(x)
        # Apply log softmax to get log probabilities which is useful for classification
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, _: DictConfig) -> None:
        """
        A Convolutional Neural Network designed for the Fashion-MNIST dataset.
        Note: The configuration parameter is not used here but provided to keep a consistent interface.

        Args:
            _ (DictConfig): The configuration object passed to the model (unused).
        """
        super(CNNFashion_Mnist, self).__init__()
        # Define the first convolutional block with Batch Normalization and Max Pooling
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Define the second convolutional block with Batch Normalization and Max Pooling
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Define the fully connected layer for classification
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the neural network using defined layers.

        Args:
            x (torch.Tensor): The input data tensor.

        Returns:
            torch.Tensor: The output tensor produced by fully connected layer.
        """
        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)
        # Flatten the output tensor to fit into the fully connected layer
        out = out.view(out.size(0), -1)
        # Apply the fully connected layer
        out = self.fc(out)
        return out
