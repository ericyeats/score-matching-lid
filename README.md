# score-matching-lid
Demo code for "A Connection Between Score Matching and Local Intrinsic Dimension" NeurIPS 2025 SPIGM

https://arxiv.org/abs/2510.12975

## Installation

We recommend you create a virtual environment. We used conda with Python 3.10, but more recent versions should be fine:

`conda create -n sm-lid python=3.10`

`conda activate sm-lid`

Install `torch`, `scikit-dimension`, `numpy`, `scikit-learn`, and `matplotlib`.

## Running the Benchmark

We provided a simple Python script `benchmark.py` which compares the performance of parametric and non-parametric estimators on a set of manifolds. You can run it with a command like this:

`python benchmark.py result_output.json --arch mlp --nonparametric_n_neighbors 50 --parametric_sigmas 0.01 0.02 0.05 --parametric_n_samples 8`

The output will be saved in `result_output.json` which associates each LID estimator and hyperparameter combination with a list of metric dictionaries. Each metric dictionary is reported from one manifold in the following order: hypersphere, hyperball, hypertwinpeaks, clifford torus, and nonlinear.

## Demo iPyNotebook

Please see `geometry.ipynb` for a simple walkthrough of how to use the `denoiser_geometry` supporting package.

## Citation

If you found this code useful, please cite:

```
@article{yeats2025connection,
  title={A Connection Between Score Matching and Local Intrinsic Dimension},
  author={Yeats, Eric and Jacobson, Aaron and Hannan, Darryl and Jia, Yiran and Doster, Timothy and Kvinge, Henry and Mahan, Scott},
  journal={arXiv preprint arXiv:2510.12975},
  year={2025}
}
```