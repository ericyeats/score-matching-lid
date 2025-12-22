from argparse import ArgumentParser
from denoiser_geometry import models as dg_models
from denoiser_geometry.methods.denoiser import Denoiser
from denoiser_geometry.methods.flowmatch import FlowMatch
from denoiser_geometry.data import flipd_dists
from denoiser_geometry import lid_estimators, eval
import json

def get_mlp_model_fn(ambient_dim: int):
    h_dim = max(2*ambient_dim, 128)
    model = dg_models.mlp.DiffusionMLP(ambient_dim, hidden_dims=[h_dim] * 4, time_embed_dim=64, 
                         class_embed_dim=64, num_classes=0, use_layer_norm=True, activation='silu', dropout=0.)
    return model

def get_mlp_kamkari_model_fn(ambient_dim: int):
    h_dim = 512
    hidden_dims = [h_dim * 4, h_dim * 2, h_dim * 2, h_dim, h_dim * 2, h_dim * 2, h_dim * 4]
    model = dg_models.mlp_kamkari.DiffusionMLPKamkari(ambient_dim, hidden_dims=hidden_dims, time_embed_dim=64, 
                         class_embed_dim=64, num_classes=0, use_layer_norm=False, activation='silu', dropout=0.)
    return model

def get_dit_model_fn(ambient_dim: int):
    patch_size = 4
    h_dim = 32*patch_size
    model = dg_models.dit.DiffusionTransformer(ambient_dim, num_classes=0, embed_dim=h_dim, num_layers=3,
                                 num_heads=8, time_embed_dim=h_dim, class_embed_dim=h_dim,
                                 patch_size=patch_size)
    return model

MODEL_FUNC_REGISTRY = {
    "mlp": get_mlp_model_fn,
    "mlp_kamkari": get_mlp_kamkari_model_fn,
    "dit": get_dit_model_fn
}

TESTBENCH_N_DATA_SAMPLES = 2000

TESTBENCH_DATASETS = [
    flipd_dists.SKDimManifold("hypersphere", 64, 16),
    flipd_dists.SKDimManifold("hyperball", 64, 16),
    flipd_dists.SKDimManifold("hypertwinpeaks", 256, 128),
    flipd_dists.CliffordTorus(128, 32),
    flipd_dists.SKDimManifold("nonlinear", 128, 32),
]

TESTBENCH_PARAMETRIC_LID_ESTIMATORS = {
    "DL": lid_estimators.parametric.DenoisingLossLIDEstimator,
    "FLIPD": lid_estimators.parametric.FLIPDLIDEstimator,
}

TESTBENCH_NONPARAMETRIC_LID_ESTIMATORS = {
    "MLE": lid_estimators.nonparametric.MLELIDEstimator,
    "TwoNN": lid_estimators.nonparametric.TwoNNLIDEstimator,
    "ESS": lid_estimators.nonparametric.ESSLIDEstimator
}

TESTBENCH_PARAMETRIC_TRAIN_PARAMS = {
    "n_batches": 50000, "print_every": 1000, "learning_rate": 1e-3, "time_range": [0., 0.1], "min_lr": 1e-5, "weight_decay": 0
}

def get_parser() -> ArgumentParser:

    parser = ArgumentParser()

    parser.add_argument('output_path', type=str, help="path to output json file")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--arch', type=str, default='mlp_kamkari', choices=['mlp_kamkari', 'mlp', 'dit'])
    parser.add_argument('--nonparametric_n_neighbors', type=int, default=100)
    parser.add_argument('--parametric_sigmas', type=float, nargs='+', default=[0.01, 0.02, 0.05])
    parser.add_argument('--parametric_n_samples', type=int, default=8)


    return parser


if __name__ == "__main__":

    parser = get_parser()

    ## get arguments
    args = parser.parse_args()

    def get_denoiser_fn(ambient_dim: int) -> Denoiser:

        model = MODEL_FUNC_REGISTRY[args.arch](ambient_dim)

        denoiser = FlowMatch(
            model=model,
            ambient_dim=ambient_dim
        )

        return denoiser

    out_dict = eval.testbench.run_testbench(
        get_model_fn=get_denoiser_fn,
        dataset_n_samples=TESTBENCH_N_DATA_SAMPLES,
        n_neighbors_nonparametric=args.nonparametric_n_neighbors,
        parametric_n_samples=args.parametric_n_samples,
        testbench_datasets=TESTBENCH_DATASETS,
        parametric_t=args.parametric_sigmas,
        testbench_nonparametric_lid_estimators=TESTBENCH_NONPARAMETRIC_LID_ESTIMATORS,
        testbench_parametric_lid_estimators=TESTBENCH_PARAMETRIC_LID_ESTIMATORS,
        normalize=True,
        parametric_train_params=TESTBENCH_PARAMETRIC_TRAIN_PARAMS,
        device=args.device
    )

    # Save with nice formatting
    with open(args.output_path, 'w') as f:
        json.dump(out_dict, f, indent=4, sort_keys=True)

    print("Saved Results to:", args.output_path)

    exit(0)

