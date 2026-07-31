# PI-DeepONet

This repository contains the code for training a Physics Informed DeepONet on the 1D slab considered in [Sahadath et al](https://www.tandfonline.com/doi/full/10.1080/00295639.2025.2586955). The models are built using Jax, building on the architecture presented by [Wang et al](https://www.science.org/doi/10.1126/sciadv.abi8605), whose code is available on [GitHub](https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets).

## Models

Four model variants are implemented across three files:

- **`nonPI_model.py`** — the `DeepONet` class: a plain (non physics-informed) model that predicts the scalar flux `phi_0(x)` directly. The same class, at two different architecture sizes, gives both the "Benchmark" checkpoint (small, matching Sahadath et al.'s architecture) and the "Large DeepONet" checkpoint (scaled up to match the PI models' parameter count, to check that the PI models don't just win by having more parameters); `nonPI_training.py` currently builds the large-architecture version.
- **`model.py`** — defines `PI_DeepONet`, which provides the physics-informed training machinery shared by every PI model. Its own trunk takes `(x, mu)` jointly, predicting the angular flux `psi(x, mu)` at a single point per forward pass. The file also defines `PI_DeepONet_Angular(PI_DeepONet)`, whose trunk instead takes `x` only, with an output layer sized `A*p` reshaped into an `(A, p)` matrix — so a single forward pass (`angular_net`) yields the angular flux at *all* 16 angles at once via a matrix product with the branch output; `operator_net` is overridden to pick a single angle's value out of that vector when the inherited `loss_bcs`/`loss_res`/etc. need one. This vector-output architecture, not the plain joint-trunk `PI_DeepONet`, is what actually produces both "PI DeepONet 1" and "PI DeepONet 2" from the report (see below) — `PI_DeepONet` itself is trainable via `training.py`, but that checkpoint isn't currently one of the models compared in `compare_models.py`.
- **`model_angular_scalar.py`** — `PI_DeepONet_AngularScalar(PI_DeepONet_Angular)`, "PI DeepONet 1": keeps `PI_DeepONet_Angular`'s vector-output trunk architecture unchanged, but overrides `loss_data` back to `PI_DeepONet`'s scalar `phi_0`-via-quadrature form instead of `PI_DeepONet_Angular`'s vector-MSE-on-`psi` form. So "PI DeepONet 1" and "PI DeepONet 2" (`PI_DeepONet_Angular` directly) share the exact same architecture and differ only in which labels supervise `loss_data` — scalar `phi_0` vs. full angular `psi`.

All PI models are trained on a weighted sum of `loss_data` (supervised scalar/angular flux fit), `loss_res` (the 1D transport PDE residual), and `loss_bcs` (the vacuum boundary condition).

## Data Generation

- **`data_generator.py`** — generates the training datasets. `sample_grf` / `squared_exp_kernel` draw source functions `Q(x)` from a Gaussian random field; `solve_sn_1d` solves the 1D transport equation via discrete ordinates with Gauss-Legendre quadrature, as in Sahadath et al. `generate_dataset` ties the two together and writes `datasets/<size>/M_Iso_train.npz` for the small/medium/large training sets.
- **`test_data_generator.py`** — generates the Sahadath test scenarios (no-shift, small-shift, large-shift GRF parameters, linear/nonlinear source combinations, and sinusoidal sources) into `datasets/test/`.
- **`validation_data_generator.py`** — generates the held-out validation set (`datasets/M_Iso_val.npz`) used for early stopping / best-params tracking during training.

## Training

- **`angular_training.py`** — trains `PI_DeepONet_Angular` on angular flux labels; produces "PI DeepONet 2".
- **`angular_scalar_training.py`** — trains `PI_DeepONet_AngularScalar` on scalar flux labels; produces "PI DeepONet 1".
- **`nonPI_training.py`** — trains `DeepONet`; produces the "Benchmark" and "Large DeepONet" checkpoints (depending on which architecture is configured in the script).
- **`training.py`** — trains the plain `PI_DeepONet`. Not currently one of the models compared elsewhere in the repo.
- **`optimization.py`** — Optuna hyperparameter search over branch/trunk width, depth, activation function, and the data/residual/BC loss weights.

Each training script picks a dataset size (`small` / `medium` / `large`), builds the training/validation batches via the `build_*_arrays` helpers, trains with early-stopping on validation ARE, and writes the resulting parameters plus training logs to a `.pkl` checkpoint under `trained_models/<size>/`.

## Evaluation and Plotting

- **`helpers.py`** — `load_model` reconstructs the correct model class from a checkpoint's saved config, plus the `r2_score` and `are` (average relative error) metrics used everywhere else.
- **`compare_models.py`**, **`compare_medium_data_models.py`**, **`compare_small_data_models.py`** — evaluate a hand-picked set of checkpoints (edit the `checkpoints`/`MODELS` dict at the top of each) across the test scenarios and plot ARE/R2 comparisons or side-by-side sample predictions.
- **`plot.py`** — plots true-vs-predicted scalar flux (with the source `Q(x)` on a secondary axis) for a single model across several test scenarios.

## Repository layout

- `datasets/<size>/` — training data; `datasets/test/` — test scenarios; `datasets/M_Iso_val.npz` — validation set.
- `trained_models/<size>/` — trained model checkpoints (`.pkl`).
- `figures/`, `results/` — output plots.
- `report.pdf` — the report.
