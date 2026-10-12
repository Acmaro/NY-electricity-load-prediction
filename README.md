# NY Day-ahead Electricity Load Forecasting

Forecast tomorrow's 5-minute electricity load (288 points) for a NYISO zone (default: Capital, `CAPITL`)
from recent load history, weather and calendar features, and compare against NYISO's own day-ahead forecast.

## Results

Day-ahead backtest: for every test day the model only sees history up to midnight of the day before
(see [Evaluation](#evaluation)). Metrics are in MW, on-the-hour points, test year 2021.

| Model | MAE | MAPE | R² |
|---|---|---|---|
| LSTM (original notebook weights, `models/legacy/LSTM.pt`) | 78.3 | 5.59 % | 0.801 |
| GRU (original notebook weights, `models/legacy/GRU.pt`) | 90.4 | 6.46 % | 0.750 |
| **NYISO day-ahead forecast** | **60.1** | **4.35 %** | **0.894** |

Reproduce with `nyload-eval data=legacy_2020 model=lstm checkpoint=models/legacy/LSTM.pt`
(on weather built with `--weather-source weather_com --station KLGA`, which is what those weights were trained on).
Numbers reported by the earlier notebooks were computed on partial periods and are not comparable.

## Setup

Python ≥ 3.10. Install the CUDA build of PyTorch first, then the package.

**Windows (native, e.g. RTX 4070 Ti)**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

**WSL2 / Linux** – same with `python -m venv .venv && source .venv/bin/activate`.
Keep the repo and data on the Linux filesystem (`~/...`), not under `/mnt/c`, or I/O becomes the bottleneck.

**macOS / CPU** – `pip install -e ".[dev]"` is enough.

**RunPod** – start a pod from a PyTorch template with a network volume mounted at `/workspace`:
```bash
cd /workspace && git clone https://github.com/Acmaro/NY-electricity-load-prediction.git
cd NY-electricity-load-prediction && bash scripts/runpod_setup.sh
source ~/.nyload_env && nyload-train env=runpod
```

## Data

Data is not stored in git; it is downloaded and rebuilt:
```bash
nyload-data all --years 2013-2023
```

| Source | Content | Raw location |
|---|---|---|
| NYISO `pal` archive | actual load, 5-min, all zones | `data/raw/nyiso/pal/` |
| NYISO `isolf` archive | NYISO hourly load forecast (benchmark) | `data/raw/nyiso/isolf/` |
| Open-Meteo archive (default) | hourly temp, dew point, humidity, wind, pressure, precip; no API key | `data/raw/weather/` |
| weather.com (`--weather-source weather_com`) | station observations; needs `WEATHER_COM_API_KEY` in `.env` | `data/raw/weather/` |

`data/processed/capitl_<year>.parquet` holds one row per 5 minutes of New York local time, exactly 288 rows per day:
off-grid NYISO readings are only used to interpolate missing points, the repeated hour on the DST fall-back day is
averaged and the missing hour in spring is interpolated. Source outages longer than one hour
(e.g. 2017: 24 days, 2021-04-01..07) are marked `load_gap=1` and excluded from training and evaluation windows.

The default weather station is `KALB` (Albany, inside the Capital zone). The original crawler used `KLGA` (New York City).

Set `NYLOAD_DATA_DIR` / `NYLOAD_OUTPUT_DIR` to keep data and runs elsewhere (RunPod: `/workspace/nyload/...`).

## Training

Configuration is composed with [Hydra](https://hydra.cc) from `configs/`:

| Group | Options |
|---|---|
| `data` | `capitl_2015_2019` (train 2015–19, test 2020), `recent_2019_2021` (train 2019–21, val 2022, test 2023), `legacy_2020` (the original notebook setup) |
| `model` | `gru`, `lstm` |
| `train` | `default`, `fast_debug` |
| `env` | `auto`, `cpu`, `local_win`, `local_wsl`, `runpod` |

```bash
nyload-train                                     # defaults: capitl_2015_2019 + gru + auto device
nyload-train env=local_win model=lstm train.epochs=50 model.hidden_size=128
nyload-train data=recent_2019_2021 data.gap=132  # history ends 13:00 the day before, like NYISO's issue time
nyload-train train=fast_debug env=cpu            # smoke test
nyload-train -m model=gru,lstm model.hidden_size=32,64,128   # grid sweep
```

Each run writes to `outputs/runs/<time>_<model>/`: the resolved config (`.hydra/`), `best.pt`, `last.pt`,
`history.json`, `preprocessor.json`, `metrics.json` and day-ahead predictions for the validation and test splits.

Re-evaluate a run:
```bash
nyload-eval run_dir=outputs/runs/<run> splits=[val,test]
```

## Evaluation

`nyload.eval.backtest` predicts every day of a split in one shot from the `seq_len` steps that end `gap` steps
before that day's midnight. Actual load of the predicted day is never an input. Metrics (`MAE`, `RMSE`, `MAPE`, `R²`)
are computed in MW after inverting the scaling, at 5-minute resolution and on the hour next to NYISO's forecast.
Weather inputs are observed values (a perfect-weather-forecast assumption).

## Adding a model

1. Implement a module mapping `(batch, seq_len, num_features)` to `(batch, pred_len)` and register it:
   ```python
   from nyload.models.registry import register

   @register("tcn")
   class TCNForecaster(nn.Module):
       def __init__(self, num_features, output_size, channels=64, levels=4): ...
   ```
2. Import it in `src/nyload/models/__init__.py`.
3. Add `configs/model/tcn.yaml` with `name: tcn` and its hyperparameters, then `nyload-train model=tcn`.

## Layout

```
configs/            Hydra configs (data / model / train / env)
src/nyload/
  data/             NYISO + weather download, cleaning to the 5-min grid, datasets and splits
  features/         calendar features, scaling
  models/           model registry, LSTM / GRU
  train/            trainer and `nyload-train`
  eval/             metrics, day-ahead backtest and `nyload-eval`
scripts/            RunPod setup
tests/              pytest suite (DST handling, split/leakage checks, models)
models/legacy/      weights from the original notebooks
notebooks/legacy/   original notebooks, kept for reference
```

```bash
pytest -q
```

## Limitations

- RNN models are prone to overfitting.
- COVID-19 significantly changed load patterns in 2020–21.
- Policy-driven changes in demand are hard to model.

## References

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Islam et al. (2022), MPE - Electrical Load Forecasting](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/2316474)
- [NYISO Load Trends and COVID-19 Impact](https://www.nyiso.com/documents/20142/2347703/2022-Climate-Impact-Study.pdf)
