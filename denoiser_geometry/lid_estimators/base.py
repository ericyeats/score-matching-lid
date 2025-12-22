import torch


class LIDEstimator:

    def estimate_lid(self, x: torch.Tensor) -> torch.Tensor:
        "base method to estimate the LID"
        raise NotImplementedError("estimate LID not implemented!")