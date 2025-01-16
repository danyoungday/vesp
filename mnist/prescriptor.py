"""
Implementation for the MNIST prescriptor.
"""
import torch

from deepnn import DeepNNPrescriptor, DeepNNPrescriptorFactory


class MNISTPrescriptor(DeepNNPrescriptor):
    """
    Binary classifier for the MNIST problem. Just attaches a sigmoid to the end.
    """
    def forward(self, context):
        return torch.sigmoid(super().forward(context))
    

class MNISTPrescriptorFactory(DeepNNPrescriptorFactory):
    """
    Prescriptor factory for our MNISTPrescriptor
    """
    def __init__(self, model_params: dict, device: str = "cpu"):
        super().__init__(MNISTPrescriptor, model_params, device)
