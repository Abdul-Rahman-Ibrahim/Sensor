import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CNN(nn.Module):

    def __init__(self,
                 input_shape: Tuple[int, int],
                 conv_sizes: List[Tuple[int, int, int, int]],
                 pool_sizes: List[int],
                 fc_sizes: List[int],
                 in_channel: int = 1
                 ) -> None:
        """
        Parameters:
            input_shape (Tuple[int, int]): The height and width of the input image or instance.

            conv_sizes (List[Tuple[int, int, int, int]]): A list containing N tuples, where N is the number of convolutional layers.
                Each tuple contains four integer elements in the following order:
                    - kernel_size: The size of the convolutional kernel (assuming k_h = k_w).
                    - padding_size: The size of the padding (assuming p_h = p_w).
                    - stride: The stride of the convolution (assuming stride_h = stride_w).
                    - out_channel: The number of output channels.

            pool_sizes (List[int]): A list containing N integers, where N is the number of pooling layers.
                Each integer represents the kernel size of the corresponding pooling layer (assuming k_h = k_w).
                Note: The number of pooling layers must equal the number of convolutional layers.
                The stride and padding for pooling layers are both set to 0.

            fc_sizes (List[int]): A list containing M integers, where M is the number of fully connected layers.
                Each integer in the list specifies the number of nodes in the respective fully connected layer.

            in_channel (int): The number of channels in the input image or instance.
        """

        super().__init__()

        self.input_shape = input_shape
        self.pool_sizes = pool_sizes

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        current_shape = input_shape

        for i in range(len(conv_sizes)):
            kernel_size, padding, stride, out_channel = conv_sizes[i]
            self.conv_layers.append(nn.Conv2d(
                in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=pool_sizes[i]))
            in_channel = out_channel

            conv_output_size = CNN.output_size_after_conv(
                current_shape, kernel_size, stride, padding)
            if pool_sizes[i] == 0:
                current_shape = conv_output_size
                continue
            pool_output_size = CNN.output_size_after_pool(
                conv_output_size, pool_sizes[i])
            current_shape = pool_output_size

        self.conv_final_size = in_channel * current_shape[0] * current_shape[1]

        self.fc_layers = nn.ModuleList()
        prev_size = self.conv_final_size
        for fc_size in fc_sizes:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            prev_size = fc_size

        self.num_conv_layers = len(conv_sizes)
        self.num_fc_layers = len(fc_sizes)

        self.losses = []
        self.accuracies = []

    def forward(self, x):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = F.relu(x)
            if self.pool_sizes[i] == 0:
                continue
            x = self.pool_layers[i](x)

        x = x.view(-1, self.conv_final_size)
        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
        return x

    def train_with_test(self, train_data, epochs, batch_size, eta, test_data):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=eta)

        for epoch in range(epochs):
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=batch_size, shuffle=True)
            for X, labels in train_loader:
                optimizer.zero_grad()
                y = self(X)
                loss = criterion(y, labels)
                loss.backward()
                optimizer.step()

            correct = 0
            total = len(test_data)
            with torch.no_grad():
                for X, labels in test_loader:
                    y = self(X)
                    predictions = torch.argmax(y, dim=1)
                    correct += torch.sum((predictions == labels).float())
            self.accuracies.append(correct/total)
            print('Epoch {0} Test accuracy: {1}'.format(epoch, correct/total))

    def train_with_loss(self, train_data, epochs, batch_size, eta):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=eta)

        for epoch in range(epochs):
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, shuffle=True)
            losses = 0
            for X, labels in train_loader:
                optimizer.zero_grad()
                y = self(X)
                loss = criterion(y, labels)
                loss.backward()
                optimizer.step()
                losses += loss.item()
            self.losses.append(losses/len(train_data))
            print(f'Epoch {epoch} Loss: {losses/batch_size}')

    def train(self, train_data, epochs, batch_size, eta, test_data=None):
        if test_data:
            self.train_with_test(train_data, epochs,
                                 batch_size, eta, test_data)
        else:
            self.train_with_loss(train_data, epochs, batch_size, eta)

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def output_size_after_conv(input_size, kernel_size, stride, padding):
        h, w = input_size
        kw = kernel_size
        kh = kw
        oh = (h + 2 * padding - kh) // stride + 1
        ow = (w + 2 * padding - kw) // stride + 1
        return (oh, ow)

    @staticmethod
    def output_size_after_pool(input_size, pool_size):
        h, w = input_size
        ph = pool_size
        pw = ph
        oh = h // ph
        ow = w // pw
        return (oh, ow)
