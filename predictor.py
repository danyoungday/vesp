import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
    

class CNNEncoder(torch.nn.Module):
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

        conv_layers.append(torch.nn.Flatten())
        self.encoder = torch.nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.encoder(x)


class MNISTPredictor(torch.nn.Module):
    def __init__(self, cnn_params: dict, decoder_params: dict):
        super().__init__()
        self.encoder = CNNEncoder(cnn_params)
        
        layers = []
        hidden_sizes = decoder_params["hidden_sizes"]
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], 1))
        layers.append(torch.nn.Sigmoid())

        self.decoder = torch.nn.Sequential(*layers)

        print(self.encoder)
        print(self.decoder)

    def encode(self, context):
        """
        Encodes the context with our CNN
        """
        return self.encoder(context.unsqueeze(1))

    def decode(self, encoded_context, actions):
        """
        Decodes the encoded context and actions into outcomes
        """
        concatenated = torch.cat([encoded_context, actions.unsqueeze(1)], dim=1)
        return self.decoder(concatenated)

    def forward(self, context, actions):
        """
        Does a full forward pass of the encoder then decoder.
        """
        encoded = self.encode(context)
        return self.decode(encoded, actions)


def validate(predictor: torch.nn.Module, dataset: Dataset, device: str = "cuda:0"):
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    predictor.to(device)
    predictor.eval()
    correct = 0
    for context, actions, outcomes in dataloader:
        context, actions, outcomes = context.to(device), actions.to(device), outcomes.to(device)
        pred = (predictor(context, actions) > 0.5).int()
        correct += torch.sum(pred == outcomes.unsqueeze(1)).item()
    return correct / len(dataset)


def train_model(predictor: torch.nn.Module,
                train_ds: Dataset,
                val_ds: Dataset,
                epochs: int,
                batch_size: int,
                device: str = "cuda:0"):
    optimizer = torch.optim.AdamW(predictor.parameters())
    loss_fn = torch.nn.BCELoss()

    dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    predictor.to(device)

    with tqdm(range(epochs), desc="Training model") as pbar:
        for _ in pbar:
            predictor.train()
            total_loss = 0
            for context, actions, outcomes in dataloader:
                context, actions, outcomes = context.to(device), actions.to(device), outcomes.to(device)
                optimizer.zero_grad()
                pred = predictor(context, actions)
                loss = loss_fn(pred, outcomes.unsqueeze(1))
                total_loss += loss.item() * len(context)
                loss.backward()
                optimizer.step()
            
            acc = validate(predictor, val_ds)

            pbar.set_postfix({"loss": total_loss / len(train_ds), "acc": acc})
