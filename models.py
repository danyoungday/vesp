"""
Some base PyTorch models used in other places.
"""
import torch

class CNNEncoder(torch.nn.Module):
    """
    CNN encoder for images.
    """
    def __init__(self, model_params: dict):
        super().__init__()
        conv_blocks = model_params["conv_blocks"]
        conv_layers = []
        for i, conv in enumerate(conv_blocks):
            conv_layers.append(torch.nn.Conv2d(**conv))
            if i < len(conv_blocks) - 1:
                conv_layers.append(torch.nn.ReLU())
                conv_layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
                conv_layers.append(torch.nn.BatchNorm2d(num_features=conv["out_channels"]))

        self.encoder = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        """
        Runs x through model.
        """
        return self.encoder(x)


class FCN(torch.nn.Module):
    """
    Fully connected neural network implementation.
    """
    def __init__(self, model_params: dict):
        super().__init__()
        hidden_sizes = model_params["hidden_sizes"]
        activation = torch.nn.Tanh if model_params["activation"] == "tanh" else torch.nn.ReLU
        layers = [torch.nn.Linear(model_params["in_size"], hidden_sizes[0])]

        for i in range(1, len(hidden_sizes)):
            layers.append(activation())
            layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        layers.append(activation())
        layers.append(torch.nn.Linear(hidden_sizes[-1], model_params["out_size"]))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Runs x through model.
        """
        return self.model(x)
