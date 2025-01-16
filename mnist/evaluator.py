"""
Evaluator implementation for MNIST problem.
"""
from presp.evaluator import Evaluator
import torch
from torch.utils.data import DataLoader, Subset

from mnist.dataset import MNISTDataset
from mnist.predictor import MNISTPredictor, train_model
from mnist.prescriptor import DeepNNPrescriptor


class MNISTEvaluator(Evaluator):
    """
    Evaluator for MNIST problem. We hard-code which subsets of the dataset to use for training/evaluation for
    reproducibility.
    """
    def __init__(self, predictor_params: dict, batch_size: int, device: str = "cpu"):
        super().__init__(["acc"])

        self.batch_size = batch_size

        dataset = MNISTDataset()
        train_indices = list(range(int(0.8 * len(dataset))))
        test_indices = list(range(int(0.8 * len(dataset)), len(dataset)))
        train_ds, test_ds = Subset(dataset, train_indices), Subset(dataset, test_indices)

        torch.manual_seed(42)
        self.predictor = MNISTPredictor(predictor_params["conv_params"], predictor_params["decoder_params"])
        train_model(self.predictor,
                    train_ds,
                    test_ds,
                    predictor_params["epochs"],
                    predictor_params["batch_size"],
                    device)
        self.predictor.eval()
        torch.save(self.predictor, predictor_params["save_path"])

        encoded_ds = MNISTDataset.get_encoded_ds(self.predictor, device)
        self.encoded_train_ds = Subset(encoded_ds, train_indices)
        self.device = device

    def update_predictor(self, elites: list):
        pass

    def evaluate_candidate(self, candidate: DeepNNPrescriptor):
        """
        Uses the encoded context to get actions from the candidate then passes those through the predictor.
        """
        dataloader = DataLoader(self.encoded_train_ds, batch_size=self.batch_size, shuffle=False)
        total_prob = 0
        with torch.no_grad():
            for encoded_context, _, _ in dataloader:
                encoded_context = encoded_context.to(self.device)
                actions = candidate.forward(encoded_context).squeeze()
                actions = (actions > 0.5).float()
                pred = self.predictor.decode(encoded_context, actions)
                total_prob += torch.sum(pred).item()

        return [-1 * total_prob / len(self.encoded_train_ds)]
