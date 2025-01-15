from presp.evaluator import Evaluator
import torch
from torch.utils.data import Dataset, random_split, DataLoader, Subset

from dataset import MNISTDataset
from predictor import Predictor, train_model
from prescriptor import DeepNNPrescriptor


class MNISTEvaluator(Evaluator):
    """
    TODO: To speed things up we should only encode the dataset once.
    """
    def __init__(self, conv_params: dict, decoder_params: dict, device: str = "cpu"):
        super().__init__(["acc"])
        dataset = MNISTDataset()
        train_indices = list(range(int(0.8 * len(dataset))))
        test_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
        train_ds, test_ds = Subset(dataset, train_indices), Subset(dataset, test_indices)

        self.predictor = Predictor(conv_params, decoder_params)
        train_model(self.predictor, train_ds, test_ds, 2, device)
        self.predictor.eval()

        encoded_ds = dataset.get_encoded_ds(self.predictor, device)
        self.encoded_train_ds = Subset(encoded_ds, train_indices)
        self.device = device

    def update_predictor(self, elites: list):
        pass

    def evaluate_candidate(self, candidate: DeepNNPrescriptor):
        """
        Uses the encoded context to get actions from the candidate then passes those through the predictor.
        """
        dataloader = DataLoader(self.encoded_train_ds, batch_size=128, shuffle=False)
        correct = 0
        with torch.no_grad():
            for encoded_context, _, _ in dataloader:
                actions = candidate.forward(encoded_context)
                pred = self.predictor.decode(encoded_context, actions)
                pred_correct = pred[:,1]
                correct += torch.sum(pred_correct).item()

        return [-1 * correct / len(self.encoded_train_ds)]