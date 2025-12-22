import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def compute_kde_manual(data_points, x_domain, bandwidth):
    """
    Manually compute KDE using Gaussian kernels.
    
    Parameters:
    - data_points: array-like, the observed data points
    - x_domain: array-like, the domain X positions where to evaluate KDE
    - bandwidth: float, the KDE scale/bandwidth parameter
    
    Returns:
    - kde_values: array, KDE values at each position in x_domain
    """
    data_points = np.array(data_points)
    x_domain = np.array(x_domain)
    kde_values = np.zeros_like(x_domain, dtype=float)
    
    # For each domain point, sum contributions from all data points
    for i, x in enumerate(x_domain):
        # Gaussian kernel: exp(-0.5 * ((x - xi) / bandwidth)^2)
        kernel_sum = np.sum(np.exp(-0.5 * ((x - data_points) / bandwidth) ** 2))
        # Normalize by bandwidth and sqrt(2*pi) for proper Gaussian kernel
        kde_values[i] = kernel_sum / (len(data_points) * bandwidth * np.sqrt(2 * np.pi))
    
    return kde_values

def plot_samples(
    x, 
    class_idx=None, 
    title="Samples by Class", 
    figsize=(8, 6), 
    kde_sigma=0.1
):
    """
    Plot samples with class-wise color, shuffling the point order for fairness.
    Points are colored by class, but legend contains each class only once.
    """
    if class_idx is None:
        class_idx = np.ones(len(x), dtype=int) * -1
    # Convert tensors to numpy if needed
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    if hasattr(class_idx, 'detach'):
        class_idx = class_idx.detach().cpu().numpy()
    
    n_samples, feat_dim = x.shape
    unique_classes = np.unique(class_idx)
    n_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    class_to_color = {val: colors[i] for i, val in enumerate(unique_classes)}
    
    if feat_dim == 1:
        fig, ax = plt.subplots(figsize=figsize)
        x_min, x_max = x.min(), x.max()
        x_range = x_max - x_min
        x_plot = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        for class_val in unique_classes:
            mask = class_idx == class_val
            class_data = x[mask, 0]
            if len(class_data) > 1:
                density = compute_kde_manual(class_data, x_plot, bandwidth=kde_sigma)
                ax.plot(x_plot, density, color=class_to_color[class_val], linewidth=2.5, 
                        label=f'Class {class_val}')
                ax.fill_between(x_plot, density, alpha=0.3, color=class_to_color[class_val])
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    else:
        if feat_dim == 2:
            fig, ax = plt.subplots(figsize=figsize)
        elif feat_dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            raise ValueError(f"feat_dim must be 1, 2, or 3, got {feat_dim}")

        # Randomize point plotting order
        perm = np.random.permutation(n_samples)
        x_shuffled = x[perm]
        class_idx_shuffled = class_idx[perm]
        color_list = np.array([class_to_color[cl] for cl in class_idx_shuffled])
        
        if feat_dim == 2:
            pts = ax.scatter(
                x_shuffled[:, 0], x_shuffled[:, 1], 
                c=color_list, alpha=0.3, s=30
            )
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
        else: # feat_dim==3
            pts = ax.scatter(
                x_shuffled[:, 0], x_shuffled[:, 1], x_shuffled[:, 2],
                c=color_list, alpha=0.3, s=30
            )
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
        
        # Legend with one patch per class
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=class_to_color[val], label=f"Class {val}")
            for val in unique_classes
        ]
        ax.legend(handles=legend_handles, frameon=True, fancybox=True, shadow=True)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()