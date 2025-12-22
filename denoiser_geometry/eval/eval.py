import torch
from .metrics import mean_absolute_error, measure_cuda_performance, concordance_index_vectorized
from ..lid_estimators.base import LIDEstimator
from ..data.flipd_dists.lid_base import LIDDistribution

def evaluate_lid_estimation(
        lid_estimator: LIDEstimator, 
        data_dict: dict,
        device: str = 'cuda'
) -> tuple[float, float, float, float]:
    with torch.no_grad():

        data = data_dict['samples']
        lid = data_dict['lid']
        class_idxs = data_dict['idx']

        if hasattr(lid_estimator, 'model'):
            lid_estimator.model = lid_estimator.model.to(device)
            data = data.to(device)

        @measure_cuda_performance
        def estimate_lid_func(x: torch.Tensor) -> torch.Tensor:
            return lid_estimator.estimate_lid(x)
        
        lid_estimates, execution_time, peak_memory_mb = estimate_lid_func(data)

        data = data.cpu()
        lid_estimates = lid_estimates.cpu()
        lid = lid.cpu()

        mae = mean_absolute_error(lid_estimates, lid)

        conc = concordance_index_vectorized(lid_estimates, lid)

        return {
            'mae': mae,
            'concordance': conc,
            'execution_time': execution_time,
            'peak_memory_mb': peak_memory_mb
        }
        
