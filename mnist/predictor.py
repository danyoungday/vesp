"""
Predictor class for MNIST classification problem and training/validation functions.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models import CNNEncoder, FCN


class MNISTPredictor(torch.nn.Module):
    """
    Predictor for the MNIST even/odd classification problem. Encodes the context with a CNN encoder, then concatenates
    it with the actions and decodes it with a fully connected network.
    """
    def __init__(self, cnn_params: dict, decoder_params: dict):
        super().__init__()
        self.encoder = CNNEncoder(cnn_params)
        self.decoder = FCN(decoder_params)

    def encode(self, context):
        """
        Encodes the context with our CNN
        """
        return torch.flatten(self.encoder(context.unsqueeze(1)), start_dim=1)

    def decode(self, encoded_context, actions):
        """
        Decodes the encoded context and actions into outcomes
        """
        concatenated = torch.cat([encoded_context, actions.unsqueeze(1)], dim=1)
        return torch.sigmoid(self.decoder(concatenated))

    def forward(self, context, actions):
        """
        Does a full forward pass of the encoder then decoder.
        """
        encoded = self.encode(context)
        return self.decode(encoded, actions)


def validate(predictor: torch.nn.Module, dataset: Dataset, device: str = "cuda:0"):
    """
    A validation function for model training just to track how well it's doing. This is kinda cheating because
    we're validating with the test set but we're not really doing any hyperparameter tuning so it's ok.
    """
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
    """
    Trains the MNIST predictor model on our synthetic MNIST dataset. A simple binary classification problem.
    """
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
