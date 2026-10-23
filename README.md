# Conditional idiosyncratic volatility (cIVOL)

Research code for estimating conditional idiosyncratic volatility using a
broad common factor space, pooled private variance dynamics, and paired-twin
payoff integration over a 21-session horizon.

The main specification uses formation-information **rank rule A**, with an
equal-weight residual variance ratio bound of **1.25**. The package also
includes Bai-Ng and fixed-rank comparisons, simulations with known truth,
and applications to return predictability and forecast encompassing.

## Installation

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then install:

```bash
python -m pip install -c requirements-lock.txt -e ".[test,wrds]"
python -m civol doctor
```

`doctor` displays dependency versions and configured paths. The constraints
file pins the numerical dependencies used with Python 3.13.9.

The estimation commands use a CUDA GPU by default and require a CUDA-enabled
PyTorch installation. Set `CIVOL_GPU_PYTHON` if PyTorch runs in a separate Python
environment. For CPU execution, use `--backend numpy`. CPU and GPU results
can differ because the backends use different random-number generators and
precision. GPU failures fall back to NumPy and are recorded as
`backend=numpy_fallback` with a `gpu_error` field.

## Data

CRSP data requires a WRDS account with access to **`crsp.dsf_v2` (CIZ)**.
Daily Fama-French factors and the market calendar are downloaded from the
[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/data_library.html).
Raw data is not included.

```bash
# Preview the SQL query and file destinations without connecting
python -m civol download --first-year 1985 --last-year 2025 --dry-run

# Download inputs and build the daily cache
python -m civol download --first-year 1985 --last-year 2025 --username YOUR_WRDS_USERNAME
python -m civol cache --first-year 1985 --last-year 2025
```

`WRDS_USERNAME` can supply the username instead. Use `--crsp-only` or
`--factors-only` to download each source separately. Completed downloads are
reused; manifests record query parameters, retrieval times, and file hashes.
Annual extracts and the full cache can require substantial memory and storage.

The cache removes duplicate `(permno, date)` rows and applies CIZ return-quality
and previous-session filters. Formation samples require ordinary US common
equity on NYSE/AMEX/Nasdaq, 750 complete sessions, stable PERMCO, positive
capitalization, and one security per issuer (largest capitalization, then
lowest PERMNO). Data revisions can affect results; exact replication requires
matching input vintages, dependencies, runtime settings, backends, and seeds.

## Running the analyses

After building the cache, run:

```bash
# Historical application and benchmark comparisons
python -m civol application --workers 6 --threads 2 --backend gpu
python -m civol analyze-application

# Four simulation designs, eight economies each, 2,000 issuers per economy
python -m civol simulation --workers 4 --threads 3 --backend gpu
python -m civol analyze-simulation

# Rank rules A and B, plus K=20; analysis uses the application records
python -m civol rank-rule --workers 6 --threads 2 --backend gpu
python -m civol analyze-rank-rule

# Return predictability and forecast encompassing
python -m civol puzzle
python -m civol encompassing

# Calendar-month-end estimates and Ang et al. comparisons
python -m civol monthend --workers 6 --threads 2 --backend gpu
python -m civol analyze-ang anchor
python -m civol analyze-ang compare
```

Defaults are 750 formation sessions, a 21-session horizon, and 16,384 integration
pairs in eight batches. Common and private variances are bounded at 25 times
their formation scale. The historical application covers 1988-2025; the
month-end application ends in November 2025.

For a single-date run, use `application --limit 1`. This keeps the full
estimation settings. `python -m civol --help` lists the commands; each command
also supports `--help`.

## Outputs and paths

Inputs are stored in `data/`, estimation records in `work/<stage>/records/`,
and tables and figures in `work/<stage>/results/`. Each completed estimation
record contains a JSON summary and NPZ arrays. Reruns skip complete pairs;
incomplete pairs stop the runner, and failures are saved as `*.FAILED.json`.
Use a separate work directory for each configuration or data vintage.

Paths default to the current working directory. Global path options precede
the command:

```bash
python -m civol --data-dir /path/to/data --work-dir /path/to/work cache
python -m civol --cache-dir /path/to/cache --work-dir /path/to/new-work application --limit 1 --backend numpy
```

Environment variables provide the same overrides and more specific paths:

| Variable | Default |
|---|---|
| `CIVOL_DATA_DIR` | `./data` |
| `CIVOL_WORK_DIR` | `./work` |
| `CIVOL_CRSP_DIR` | `data/crsp_daily` |
| `CIVOL_FACTORS_FILE` | `data/factors_daily.csv` |
| `CIVOL_CALENDAR_FILE` | `data/experiment_ready/market_calendar.parquet` |
| `CIVOL_CACHE_DIR` | `work/cache` |
| `CIVOL_GPU_PYTHON` | Current Python executable |
| `CIVOL_APPLICATION_RECORDS` | `work/application/records` |
| `CIVOL_SIMULATION_RECORDS` | `work/simulation/records` |
| `CIVOL_RANK_RULE_RECORDS` | `work/rank_rule/records` |
| `CIVOL_ANG_COMPARISON_RECORDS` | `work/ang_comparison/records` |

Set environment variables before starting Python. Record overrides apply to
both estimation and analysis; analysis outputs follow `CIVOL_WORK_DIR`.

## Tests

```bash
python -m pytest -q
```

The tests use synthetic inputs and require no licensed data or network access.
CUDA tests run when a compatible GPU environment is available. Optional
comparisons with archived research code are skipped unless configured.
