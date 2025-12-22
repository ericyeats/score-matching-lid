from .lid_base import LIDDistribution
import torch
from skdim import datasets as dimdata
import numpy as np
from functools import partial

MANIFOLD_NAMES = ['hypersphere', 'hypertwinpeaks', 'hyperball', 'nonlinear', 'paraboloid']

def pad_zeros(data: np.ndarray, dim: int):
    # data is [n_samples, data_dim]
    assert dim >= 0
    if dim > 0:
        pad = np.zeros((data.shape[0], dim), dtype=data.dtype)
        data = np.concatenate([data, pad], axis=1)
    return data

class SKDimManifold(LIDDistribution):

    def __init__(
        self,
        manifold_name: str,
        ambient_dim: int | None = None,
        tangent_dim: int | None = None
    ):
        super().__init__()
        self.manifold_name = manifold_name
        self.ambient_dim = ambient_dim
        self.tangent_dim = tangent_dim

    def sample(
        self,
        sample_shape: tuple[int, ...] | int,
        return_dict: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        If return_dict is True, returns a dictionary with the key 'samples' and 'lid' and 'idx'
        which are the samples, the local intrinsic dimensionality of the submanifold associated
        with that data and 'idx' is the index of submanifold.
        """
        
        if seed is not None:
            torch.manual_seed(seed)

        if self.manifold_name == 'hypersphere':

            data = dimdata.hyperSphere(sample_shape, self.tangent_dim) # returns (n_samples, d+1)

        elif self.manifold_name == 'hyperball':
            
            data = dimdata.hyperBall(sample_shape, self.tangent_dim)
            
        elif self.manifold_name == 'hypertwinpeaks':

            data = dimdata.hyperTwinPeaks(sample_shape, self.tangent_dim) # returns (n_samples, d+1)

        elif self.manifold_name == 'nonlinear':

            datagen = dimdata.BenchmarkManifolds()
            data = datagen._gen_campadelli_n_data(sample_shape, dim=4*self.tangent_dim, d=self.tangent_dim)

        elif self.manifold_name == 'paraboloid':

            datagen = dimdata.BenchmarkManifolds()
            data = datagen._gen_paraboloid_data(sample_shape, dim=3*(self.tangent_dim+1), d=self.tangent_dim)

        else:
            raise ValueError(f"manifold_name {self.manifold_name} not recognized")

        data = pad_zeros(data, self.ambient_dim - (data.shape[1]))
        data = torch.from_numpy(data).float()

        if not return_dict:
            return data
        
        lids = torch.full((data.shape[0],), fill_value=self.tangent_dim, dtype=torch.long)
        idx = torch.zeros_like(lids)

        out_dict = {
            'samples': data,
            'lid': lids,
            'idx': idx
        }

        return out_dict