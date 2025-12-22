from .eval import evaluate_lid_estimation
from ..lid_estimators.nonparametric import MLELIDEstimator, TwoNNLIDEstimator, ESSLIDEstimator
from ..lid_estimators.parametric import DenoisingLossLIDEstimator, ImplicitLossLIDEstimator, EigenvalueLIDEstimator, FLIPDLIDEstimator
from ..data.flipd_dists.skdim_manifolds import SKDimManifold
from typing import Callable
from ..methods.denoiser import Denoiser
from ..models.mlp import DiffusionMLP
from ..methods.flowmatch import FlowMatch
import torch

import torch


def default_get_model_fn(ambient_dim: int):
    h_dim = 128
    model = DiffusionMLP(ambient_dim, hidden_dims=[h_dim] * 6, time_embed_dim=h_dim, class_embed_dim=h_dim, num_classes=0)
    denoiser = FlowMatch(
        model=model,
        ambient_dim=ambient_dim
    )
    return denoiser

import torch

def normalize_data(data: torch.Tensor) -> torch.Tensor:
    data = data - data.mean(dim=0, keepdim=True)
    data_std = data.std(dim=0).max().item()
    data = data / (data_std + 1e-5)
    return data

TESTBENCH_DATASETS = [
    SKDimManifold("hypersphere", 16, 8),
    SKDimManifold("hyperball", 16, 8),
    SKDimManifold("nonlinear", 128, 32)
]

TESTBENCH_NONPARAMETRIC_LID_ESTIMATORS = {
    "MLE": MLELIDEstimator,
    "TwoNN": TwoNNLIDEstimator,
    "ESS": ESSLIDEstimator
}

TESTBENCH_PARAMETRIC_LID_ESTIMATORS = {
    "DL": DenoisingLossLIDEstimator,
    "IL": ImplicitLossLIDEstimator,
    "Eig": EigenvalueLIDEstimator,
    "FLIPD": ImplicitLossLIDEstimator,
}


def run_testbench(
    get_model_fn: Callable[[int], Denoiser] = default_get_model_fn,
    dataset_n_samples: int = 3000,
    parametric_n_samples: int = 16,
    parametric_t: list[float] = [0.01, 0.02, 0.05, 0.1, 0.2],
    parametric_train_params: dict = {"n_batches": 30000, "print_every": 1000},
    n_neighbors_nonparametric: int = 100,
    normalize: bool = False,
    seed: int = 0,
    testbench_datasets: list = TESTBENCH_DATASETS,
    testbench_nonparametric_lid_estimators: dict = TESTBENCH_NONPARAMETRIC_LID_ESTIMATORS,
    testbench_parametric_lid_estimators: dict = TESTBENCH_PARAMETRIC_LID_ESTIMATORS,
    device = 'cuda'
):
    
    results = {}
    for name in testbench_nonparametric_lid_estimators.keys():
        results[name] = []
    
    for name in testbench_parametric_lid_estimators.keys():
        for t in parametric_t:
            name_t = f"{name}_{t:.2f}"
            results[name_t] = []

    print("Number of Datasets:", len(testbench_datasets))
    
    for dataset in testbench_datasets:

        print("Sampling Dataset")

        data_dict = dataset.sample(
            dataset_n_samples,
            return_dict=True,
            seed=seed
        )

        if normalize:
            data_dict['samples'] = normalize_data(data_dict['samples'].to(device)).cpu()

        for name, lid_estimator in testbench_nonparametric_lid_estimators.items():
            print(name)

            lid_estimator_inst = lid_estimator(n_neighbors=n_neighbors_nonparametric)

            eval_dict = evaluate_lid_estimation(
                lid_estimator_inst,
                data_dict
            )

            results[name].append(eval_dict)

        if len(testbench_parametric_lid_estimators):
            print("Training Model")

            torch.manual_seed(seed)

            model = get_model_fn(dataset.ambient_dim).to(device)

            # train the model
            model.train_dsm(
                data_dict['samples'].to(device),
                device=device,
                **parametric_train_params
            )

        for name, lid_estimator in testbench_parametric_lid_estimators.items():

            print(name)

            for t in parametric_t:
                name_t = f"{name}_{t:.2f}"

                estimator_kwargs = {
                    "model": model, 
                    "t": t, 
                    "class_idx": model.model.num_classes,
                    "n_samples": parametric_n_samples
                } 

                torch.manual_seed(seed)

                lid_estimator_inst = lid_estimator(**estimator_kwargs)

                eval_dict = evaluate_lid_estimation(
                    lid_estimator_inst,
                    data_dict
                )

                results[name_t].append(eval_dict)

    return results




