import functools
import torch
import time

def measure_cuda_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return result, execution_time, peak_memory_mb
    return wrapper

def mean_absolute_error(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return (predictions.float() - ground_truth.float()).abs().mean().item()

def concordance_index_vectorized(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Vectorized calculation of concordance index using PyTorch operations.
    
    Args:
        predictions (torch.Tensor): Predicted local intrinsic dimensions [N]
        ground_truth (torch.Tensor): Ground truth local intrinsic dimensions [N]
    
    Returns:
        float: Concordance index
    """
    device = predictions.device
    n = len(predictions)
    
    if n < 2:
        return 1.0

    ground_truth = ground_truth.float()
    
    # Create difference matrices
    # gt_diff[i,j] = ground_truth[i] - ground_truth[j]
    gt_diff = ground_truth.unsqueeze(0) - ground_truth.unsqueeze(1)
    pred_diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)
    
    # Only consider upper triangular part (i < j pairs)
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
    
    gt_diff_pairs = gt_diff[mask]
    pred_diff_pairs = pred_diff[mask]
    
    # Find non-tied pairs in ground truth
    non_tied_mask = gt_diff_pairs != 0
    
    if non_tied_mask.sum() == 0:
        return 1.0  # All pairs are tied
    
    # Keep only non-tied pairs
    gt_diff_comparable = gt_diff_pairs[non_tied_mask]
    pred_diff_comparable = pred_diff_pairs[non_tied_mask]
    
    # Count concordant pairs (same sign)
    concordant = (gt_diff_comparable * pred_diff_comparable > 0).float()
    
    c_index = concordant.mean().item()
    return c_index
