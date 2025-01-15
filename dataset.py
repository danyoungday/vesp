import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import v2


class CustomDS(Dataset):
    def __init__(self, context, actions, outcomes):
        self.context = context
        self.actions = actions
        self.outcomes = outcomes

    def __len__(self):
        return len(self.context)
    
    def __getitem__(self, idx):
        return self.context[idx], self.actions[idx], self.outcomes[idx]


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
        # pylint: disable=not-callable
        self.outcomes = F.one_hot(self.outcomes, num_classes=2)

        self.actions = self.actions.float()
        self.outcomes = self.outcomes.float()


    def get_encoded_ds(self, predictor, device):
        all_context = []
        for c in self.context:
            all_context.append(predictor.encode(c.to(device).unsqueeze(0)).squeeze(0))
        return CustomDS(torch.stack(all_context), self.actions, self.outcomes)

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