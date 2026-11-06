# Conditional idiosyncratic volatility (cIVOL)

Research code for estimating conditional idiosyncratic volatility over a
21-session horizon, using a common factor space, pooled private variance
dynamics, and paired-twin payoff integration.

The current estimator selects `max(K_EW, K_BaiNg)`: the equal-weight rule
(bound 1.25, search cap 40) with a Bai–Ng ICp2 minimum (search cap 20).
This prevents the selected rank from falling below Bai–Ng's; it does not
guarantee that all common variation is captured. Earlier estimators remain
available for replication.

## Install

Requires Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -c requirements-lock.txt -e ".[test,wrds]"
python -m civol doctor
```

The constraints file pins dependencies used with Python 3.13.9.
`doctor` shows dependency versions and configured paths.

## Prepare data

You need a WRDS account with access to CRSP `crsp.dsf_v2` (CIZ).
Fama–French factors and calendar dates come from the
[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/data_library.html).
Raw data is not included.

```bash
python -m civol download --first-year 1985 --last-year 2025 --username YOUR_WRDS_USERNAME
python -m civol cache --first-year 1985 --last-year 2025
```

Add `--dry-run` to preview the download, or set `WRDS_USERNAME` instead of
passing a username. Downloads are reused when complete.

The sample uses ordinary US common equity on NYSE, AMEX, and Nasdaq,
with 750 complete formation sessions and one security per issuer.

## Run

Start with one formation date:

```bash
python -m civol safeguarded-measurement --limit 1
```

Remove `--limit 1` for the full run. Defaults are 750 formation sessions,
a 21-session horizon, and 16,384 integration pairs in eight batches.
The current estimator uses CPU by default. Add `--backend gpu` for CUDA
with PyTorch; set `CIVOL_GPU_PYTHON` if PyTorch is in another environment.
GPU failures fall back to NumPy and are recorded in the output.
CPU and GPU results can differ in precision and random draws.

Earlier analyses are available through these commands:

| Analysis | Estimate | Analyze |
|---|---|---|
| Historical application | `application` | `analyze-application` |
| Simulations | `simulation` | `analyze-simulation` |
| Rank comparisons | `rank-rule` | `analyze-rank-rule` |
| Month-end / Ang comparisons | `monthend` | `analyze-ang anchor`, then `analyze-ang compare` |

Prefix each command with `python -m civol`. These estimation commands
default to GPU; use `--backend numpy` for CPU. Run the application before
the rank analysis. `puzzle` and `encompassing` provide return-predictability
and forecast-encompassing analyses. Use `--help` for command options.

## Files

Inputs go in `data/`, estimates in `work/<stage>/records/`, and analysis
tables and figures in `work/<stage>/results/`. Each estimate has a JSON
summary and NPZ arrays. Reruns skip complete records and stop at partial
records; failures are saved as `*.FAILED.json`.

Use a separate work directory for each configuration or data vintage.
Path options go before the command:

```bash
python -m civol --data-dir /path/to/data --work-dir /path/to/work cache
```

You can also set `CIVOL_DATA_DIR`, `CIVOL_WORK_DIR`, or `CIVOL_CACHE_DIR`.
For exact replication, match the input data, dependencies, settings,
backend, and seeds.

## Tests

```bash
python -m pytest -q
```

Tests use synthetic data. GPU and archived-code comparisons run only when
their environments are available.
