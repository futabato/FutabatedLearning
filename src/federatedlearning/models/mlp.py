import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int) -> None:
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return F.softmax(x, dim=1)
