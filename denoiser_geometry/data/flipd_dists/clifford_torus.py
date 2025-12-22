import numpy as np
from .lid_base import LIDDistribution
import torch

def get_clifford_torus(N, ad, td):
    """
    Generate sample points on an n-torus.

    Parameters:
    - n: Dimension of the torus (the number of circles)
    - num_samples: Number of sample points to generate

    Returns:
    - points: An array of shape (num_samples, 2*n) containing the sampled points
    """
    assert ad >= td * 2, "Clifford Torus requires that the ambient dimension is greater or equal to twice the topological dimension"

    # Initialize the array to hold the points
    points = np.zeros((N, 2 * td))

    # Generate random angles
    angles = np.random.uniform(0, 2 * np.pi, (N, td))

    # Compute the points on the n-torus
    for i in range(td):
        # Sine and cosine of the angles for each dimension
        points[:, 2*i] = np.cos(angles[:, i])
        points[:, 2*i + 1] = np.sin(angles[:, i])

    # EY here
    # append additional ambient dimensions as needed
    points = np.concatenate([points, np.zeros((N, ad - 2 * td))], axis=1)

    return points

class CliffordTorus(LIDDistribution):

    def __init__(self, ambient_dim: int, tangent_dim: int):
        super().__init__()
        self.ambient_dim = ambient_dim
        self.tangent_dim = tangent_dim
        assert ambient_dim >= tangent_dim*2 and tangent_dim % 2 == 0

    def sample(
        self,
        sample_shape: tuple[int, ...] | int,
        return_dict: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if seed:
            torch.manual_seed(seed)

        data = get_clifford_torus(
            sample_shape,
            self.ambient_dim,
            self.tangent_dim,
        )

        # permute the columns
        perm = np.random.permutation(data.shape[1])
        data = data[:, perm]

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