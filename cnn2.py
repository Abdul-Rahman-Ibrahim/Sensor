
import torch
import cnn as cnn
from typing import Tuple, List
import torch.nn as nn
import torch.nn.functional as F


# New: Custom Layers for Memristor Mapping
class MappedConv2d(nn.Conv2d):
    """A custom convolutional layer that applies conductance mapping."""
    def forward(self, input):
        # Apply the weight-to-conductance mapping here
        G = 2.0 / (1.0 + torch.exp(-self.weight))
        return F.conv2d(input, G, self.bias, self.stride, self.padding, self.dilation, self.groups)

class MappedLinear(nn.Linear):
    """A custom linear layer that applies conductance mapping."""
    def forward(self, input):
        # Apply the weight-to-conductance mapping here
        G = 2.0 / (1.0 + torch.exp(-self.weight))
        return F.linear(input, G, self.bias)

# New: MappedCNN class
class MappedCNN(cnn.CNN):
    """A CNN class with layers replaced by memristor-mapped layers."""
    def __init__(self,
                 input_shape: Tuple[int, int],
                 conv_sizes: List[Tuple[int, int, int, int]],
                 pool_sizes: List[int],
                 fc_sizes: List[int],
                 in_channel: int = 1
                 ) -> None:

        super().__init__(input_shape, conv_sizes, pool_sizes, fc_sizes, in_channel)

        # Replace standard Conv2d layers with MappedConv2d
        self.conv_layers = nn.ModuleList()
        current_in_channel = in_channel
        for i in range(len(conv_sizes)):
            kernel_size, padding, stride, out_channel = conv_sizes[i]
            self.conv_layers.append(MappedConv2d(
                current_in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride))
            current_in_channel = out_channel

        # Replace standard Linear layers with MappedLinear
        self.fc_layers = nn.ModuleList()
        prev_size = self.conv_final_size
        for fc_size in fc_sizes:
            self.fc_layers.append(MappedLinear(prev_size, fc_size))
            prev_size = fc_size