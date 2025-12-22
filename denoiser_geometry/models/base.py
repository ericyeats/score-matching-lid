import torch

class BaseDenoiserModel(torch.nn.Module):

    def __init__(self, ambient_dim, num_classes):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.num_classes = num_classes