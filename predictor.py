
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class LinearPredictor(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(in_size, out_size), torch.nn.Softmax(dim=-1))

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        return self.model(flat)


class NNPredictor(torch.nn.Module):
    def __init__(self, model_params: dict):
        super().__init__()
        hidden_sizes = model_params["hidden_sizes"]
        layers = [torch.nn.Linear(model_params["in_size"], hidden_sizes[0])]
        for i in range(1, len(hidden_sizes)):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], model_params["out_size"]))
        layers.append(torch.nn.Softmax(dim=-1))
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        return self.model(flat)
    

class CNNPredictor(torch.nn.Module):
    def __init__(self, model_params: dict):
        super().__init__()
        conv_blocks = model_params["conv_blocks"]
        conv_layers = []
        for i, conv in enumerate(conv_blocks):
            conv_layers.append(torch.nn.Conv2d(conv["in_channels"], conv["out_channels"], conv["kernel_size"]))
            if i < len(conv_blocks) - 1:
                conv_layers.append(torch.nn.ReLU())
                conv_layers.append(torch.nn.MaxPool2d(kernel_size=2))

        conv_layers.append(torch.nn.Flatten())
        self.encoder = torch.nn.Sequential(*conv_layers)

        decoder_layers = []
        hidden_sizes = model_params["hidden_sizes"]
        for i in range(len(hidden_sizes)-1):
            decoder_layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:
                decoder_layers.append(torch.nn.ReLU())
        decoder_layers.append(torch.nn.Linear(hidden_sizes[-1], model_params["out_size"]))
        decoder_layers.append(torch.nn.Softmax(dim=-1))

        self.decoder = torch.nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x.unsqueeze(1))
        return self.decoder(encoded)


class Predictor(torch.nn.Module):
    def __init__(self, cnn_params: dict, decoder_params: dict):
        super().__init__()
        self.encoder = CNNPredictor(cnn_params).encoder
        
        layers = []
        hidden_sizes = decoder_params["hidden_sizes"]
        for i in range(len(hidden_sizes)-1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < len(hidden_sizes) - 2:
                layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_sizes[-1], decoder_params["out_size"]))
        layers.append(torch.nn.Softmax(dim=-1))

        self.decoder = torch.nn.Sequential(*layers)

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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictor.to(device)
    predictor.eval()
    correct = 0
    for context, actions, outcomes in dataloader:
        context, actions, outcomes = context.to(device), actions.to(device), outcomes.to(device)
        pred = torch.argmax(predictor(context, actions), dim=1)
        true = torch.argmax(outcomes, dim=1)
        correct += torch.sum(pred == true).item()
    return correct / len(dataset)


def train_model(predictor: torch.nn.Module, train_ds: Dataset, val_ds: Dataset, epochs: int, device: str = "cuda:0"):
    optimizer = torch.optim.AdamW(predictor.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    predictor.to(device)

    with tqdm(range(epochs), desc="Training model") as pbar:
        for _ in pbar:
            predictor.train()
            total_loss = 0
            for context, actions, outcomes in dataloader:
                context, actions, outcomes = context.to(device), actions.to(device), outcomes.to(device)
                optimizer.zero_grad()
                pred = predictor(context, actions)
                loss = loss_fn(pred, outcomes)
                total_loss += loss.item() * len(context)
                loss.backward()
                optimizer.step()
            
            acc = validate(predictor, val_ds)

            pbar.set_postfix({"loss": total_loss / len(train_ds), "acc": acc})

def test():
    torch.manual_seed(42)
    from dataset import MNISTDataset, EvenOddDataset
    dataset = MNISTDataset()
    train_ds, test_ds = random_split(dataset, [0.8, 0.2])
    # linear_predictor = LinearPredictor(28 * 28, 2)
    # nn_predictor = NNPredictor({"in_size": 28*28, "hidden_sizes": [128, 64, 32], "out_size": 2})
    conv_params = {
        "conv_blocks": [
            {"in_channels": 1, "out_channels": 16, "kernel_size": 3},
            {"in_channels": 16, "out_channels": 32, "kernel_size": 3},
        ],
        "hidden_sizes": [3872, 128, 64],
        "out_size": 2
    }
    decoder_params = {
        "hidden_sizes": [3873, 128, 64],
        "out_size": 2
    }
    # cnn_predictor = CNNPredictor(conv_params)
    predictor = Predictor(conv_params, decoder_params)

    train_model(predictor, train_ds, test_ds, 10)

if __name__ == "__main__":
    test()