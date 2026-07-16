# TTE-TAI-PROCI paper

Research code for the Proceedings of the Combustion Institute paper *Deep learning for continuous lead-time prediction of thermoacoustic instabilities in an annular combustor*. The study formulates thermoacoustic-instability forecasting as continuous regression of the remaining time to onset from multichannel pressure measurements. It compares a causal Transformer, a unidirectional LSTM, and an XGBoost feature baseline trained on synthetic trajectories from a stochastic reduced-order model of a 12-burner annular combustor.

**The datasets and pretrained checkpoints are not included in this repository.**

## Study context

The paper evaluates predictions over the final 2 seconds before detected instability onset. Synthetic pressure signals are generated at 51.2 kHz and decimated to 5.12 kHz. The 1,103 accepted synthetic trajectories are split into 744 training, 259 validation, and 100 test trajectories. The results obtained with 10 training seeds, 1337 through 1346, are compared.

Zero-shot experimental evaluation uses 13 upward ramps from a 6 kW annular combustor: four 5-second ramps, four 20-second ramps, and five 60-second ramps. For the sequence models, three pressure sensors at azimuthal positions 0, 120, and 240 degrees are embedded in a fixed 12-position input grid.

The Transformer and LSTM operate on minimally processed pressure windows. The XGBoost comparison instead uses variance, lag-5 autocorrelation, kurtosis, and azimuthal mode-1 energy fraction extracted from one-second envelope segments.

## Repository contents

| File | Purpose |
| --- | --- |
| [`synthetic_gen.py`](synthetic_gen.py) | Generates synthetic combustor trajectories and detects instability onset. |
| [`label_exp_onset.py`](label_exp_onset.py) | Labels experimental ramps with the threshold-persistence onset detector. |
| [`trans_train.py`](trans_train.py) | Trains and evaluates the causal Transformer regressor. |
| [`lstm_train.py`](lstm_train.py) | Trains and evaluates the LSTM baseline. |
| [`XGBoost_train.py`](XGBoost_train.py) | Extracts envelope-precursor features from synthetic trajectories and trains the XGBoost baseline. |
| [`model.py`](model.py) | Defines the Transformer and LSTM models. |
| [`preprocess_mod.py`](preprocess_mod.py) | Streams normalized pre-onset windows from saved trajectories. |
| [`common.py`](common.py) | Provides shared data, onset-detection, reproducibility, and serialization utilities. |

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- PyTorch
- tqdm
- pandas
- scikit-learn
- XGBoost
- A CUDA-capable GPU for the default training configurations; CPU execution is supported with `--device cpu` but is substantially slower

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy torch tqdm pandas scikit-learn xgboost
```

## Expected data layout

Synthetic run files must contain a two-dimensional tensor with shape `[time, 12 sensors]`. Each matching metadata file must contain an `onset_ts` entry expressed as a sample index.

Experimental run files must contain a two-dimensional tensor with shape `[time, 3 sensors]`. They are decimated by a factor of 10 when loaded. The experimental-onset JSON maps run IDs to onset indices in the decimated signals:

```json
{
  "0": 24500,
  "1": 23120
}
```

## Workflow

### 1. Generate synthetic trajectories

```bash
python synthetic_gen.py \
  -n_sim 1200 \
  -out_dir data \
  -T 10 \
  -dt 1.953125e-05 \
  -n_sens 12
```

The generator simulates coupled stochastic burner amplitudes, reconstructs noisy sensor signals, decimates them by a factor of 10, and retains trajectories for which a valid onset is detected.

The output subdirectories are currently fixed in the generator:

- `data/runs1/sim_<id>.pt`: sensor trajectories
- `data/dct_meta1/sim_<id>.pth`: simulation metadata and `onset_ts`

Rejected simulations do not produce files, so the number of saved trajectories can be smaller than `-n_sim`.

### 2. Label experimental onsets

Place the raw experimental tensors in `data/experimental/` using the name `ramp_<id>.pt`, then run:

```bash
python label_exp_onset.py \
  --exp-dir data/experimental \
  --out-json data/experimental_onsets.json \
  --workers 4
```

The default detector uses a quantile-derived amplitude threshold followed by persistence and local-standard-deviation checks. Its main parameters are defined through `--onset-thresh`, `--pers-quantile`, `--pers-adjust`, and `--persistence-fraction`.

### 3. Train the Transformer

```bash
python trans_train.py \
  --meta-dir data/dct_meta1 \
  --runs-dir data/runs1 \
  --exp-run-dir data/experimental \
  --exp-onset-json data/experimental_onsets.json \
  --out-dir outputs/transformer \
  --tag transformer_seed1337 \
  --device cuda:0 \
  --seed 1337 \
  --n-traj-train 744 \
  --n-traj-burst-val 259 \
  --test-size 100
```

The model receives a normalized pressure window and a sensor-availability mask. Training uses:

- three active sensors,
- log-transformed time-to-onset targets,
- deterministic seeding,
- validation-based checkpointing,
- synthetic test evaluation when a test split exists,
- transfer evaluation on the experimental ramps.

Use `--init-checkpoint <path>` to initialize the Transformer from an existing checkpoint. Use `--amp-train` to enable CUDA BF16 autocasting during training.

### 4. Train the LSTM baseline

```bash
python lstm_train.py \
  --meta-dir data/dct_meta1 \
  --runs-dir data/runs1 \
  --exp-run-dir data/experimental \
  --exp-onset-json data/experimental_onsets.json \
  --out-dir outputs/lstm \
  --tag lstm_seed1337 \
  --device cuda:0 \
  --seed 1337 \
  --n-traj-train 744 \
  --n-traj-burst-val 259 \
  --test-size 100
```

The LSTM uses the same window sampling, sensor masking, label normalization, data splitting, and experimental-transfer evaluation as the Transformer.

### 5. Train the XGBoost baseline

`XGBoost_train.py` performs the complete synthetic XGBoost pipeline in one run:

1. Optionally relabels the synthetic onset metadata with the configured threshold-persistence detector.
2. Splits the available trajectories into training and validation sets.
3. Samples pre-onset windows from each trajectory.
4. Selects synthetic sensor channels 0, 4, and 8 to match the three experimental azimuths.
5. Estimates the carrier frequency, demodulates the three pressure signals, low-pass filters their complex envelopes, and resamples them to 160 Hz.
6. Extracts variance, lag-5 autocorrelation, kurtosis, and mode-1 energy fraction from each one-second feature window.
7. Fits an `XGBRegressor` to the log time-to-onset target and evaluates it on the validation split.

The script reads 12-channel trajectories from `Config.run_dir` and matching metadata from `Config.meta_dir`. Its current defaults are `data/runs3` and `data/dct_meta3`; change them to `data/runs1` and `data/dct_meta1` when using the output of `synthetic_gen.py` shown above.

Run the pipeline with:

```bash
python XGBoost_train.py
```

There is no command-line interface. Configure the run by editing `SEED` and the `Config` dataclass in `XGBoost_train.py`. Important defaults include:

- seed `1343` with split seed `0`;
- 900 training trajectories, with all remaining trajectories used for validation;
- 64 sampled windows per training and validation trajectory;
- a maximum lead-time buffer of 10,240 samples;
- 1,200 histogram-based trees with learning rate `0.03` and 50-round early stopping;
- output directory `xgb_env_precursors_norminput_3sens_<seed>`.

The generated feature tables contain trajectory ID, window endpoint, distance to onset, log distance to onset, and the four handcrafted precursor columns.

## Sequence-model outputs

Each run writes files under its `--out-dir`:

- `<tag>_summary.json`: configuration, split sizes, training history, test metrics, and experimental-transfer metrics
- `<tag>_monitor.json`: compact current-state information for monitoring a running job
- one or more `.pth` checkpoints containing the model state, label-normalization statistics, configuration, and run tag

The primary reported time-domain metric is mean absolute error in seconds. Predictions are converted from standardized log time steps using the post-decimation interval `DT = 1.953125e-4` seconds.

## XGBoost outputs

`XGBoost_train.py` writes these artifacts under `Config.out_dir`:

- `xgb_env_precursors_3sens_model.json`: serialized XGBoost model
- `train_features_env_precursors_3sens.csv`: generated training feature table
- `val_features_env_precursors_3sens.csv`: generated validation feature table
- `metrics_train.json`: training MAE, MSE, and RMSE in log units and seconds
- `metrics_val.json`: validation MAE, MSE, and RMSE in log units and seconds
- `config.json`: effective configuration, metadata source, and onset-labelling information
- `feature_columns.json`: ordered feature list used by the model

## Useful options

Inspect every supported command-line option with:

```bash
python synthetic_gen.py --help
python label_exp_onset.py --help
python trans_train.py --help
python lstm_train.py --help
```

`XGBoost_train.py` is configured by editing `SEED` and its `Config` dataclass rather than command-line options.

The full training defaults assume a large synthetic dataset. For smaller datasets, set `--n-traj-train`, `--n-traj-burst-val`, and `--test-size` so that the requested split does not exceed the number of valid saved trajectories.

## Reproducibility notes

- Synthetic generation uses a fixed top-level seed of `0`; each trajectory receives a derived seed.
- Model initialization and sampling use the value supplied through `--seed`.
- The train/validation/test trajectory split uses a fixed seed of `0`.
- PyTorch deterministic algorithms are enabled where supported.
- Training summaries persist the effective configuration and trajectory-pool sizes.
- XGBoost seeds Python, NumPy, PyTorch, window sampling, and model fitting from the module-level `SEED`; its trajectory split uses `Config.split_seed`.
- The XGBoost configuration and effective onset-metadata source are saved with every run.
