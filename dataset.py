import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2


class MNISTDataset(Dataset):
    """
    A synthetic dataset representing decision making based on MNIST images.
    The synthetic outcome is whether a theoretical agent guessed right or wrong on whether or not the digit is even.
    The actions are therefore generated as the correct label for the image, then are flipped if the outcome is incorrect
    """
    def __init__(self):
        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms)

        self.context = self.dataset.data / 255.0
        self.actions = self.dataset.targets % 2
        self.outcomes = torch.randint(0, 2, self.actions.shape)
        self.actions[~self.outcomes.bool()] = 1 - self.actions[~self.outcomes.bool()]

        self.actions = self.actions.float()
        self.outcomes = self.outcomes.float()

    @staticmethod
    def encode_context(context, predictor, device):
        all_context = []
        for c in context:
            all_context.append(predictor.encode(c.to(device).unsqueeze(0)).squeeze(0))
        return torch.stack(all_context)

    @classmethod
    def get_encoded_ds(cls, predictor, device):
        ds = cls()
        ds.context = ds.encode_context(ds.context, predictor, device) 
        return ds
    
    @classmethod
    def get_encoded_eval_ds(cls, predictor, device):
        ds = cls()
        ds.context = ds.encode_context(ds.context, predictor, device)
        eval_actions = ds.dataset.targets % 2
        eval_outcomes = torch.ones_like(eval_actions)
        # pylint: disable=not-callable
        ds.actions = eval_actions.float()
        ds.outcomes = eval_outcomes.float()
        return ds
    
    @classmethod
    def get_counterfactual_ds(cls, predictor, device):
        ds = cls()
        ds.context = ds.encode_context(ds.context, predictor, device)
        ds.actions = 1- ds.actions
        ds.outcomes = 1 - ds.outcomes
        return ds

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        return self.context[idx], self.actions[idx], self.outcomes[idx]


class BaseMNISTDataset(Dataset):
    def __init__(self):
        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms)

        self.imgs = self.dataset.data / 255.0

        self.targets = self.dataset.targets
        # pylint: disable=not-callable
        self.targets = F.one_hot(self.targets, num_classes=10).float()

        print(self.imgs.shape, self.targets.shape)
        print(self.imgs.max(), self.imgs.min())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]


class EvenOddDataset(Dataset):
    def __init__(self):
        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms)

        self.imgs = self.dataset.data / 255.0

        self.targets = self.dataset.targets % 2
        # pylint: disable=not-callable
        self.targets = F.one_hot(self.targets, num_classes=2).float()

        print(self.imgs.shape, self.targets.shape)
        print(self.imgs.max(), self.imgs.min())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.imgs[idx], self.targets[idx]