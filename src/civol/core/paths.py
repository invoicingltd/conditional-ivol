"""Portable locations, configured through environment variables before import.

Input data and generated work are deliberately separate from installed source.
Relative overrides are resolved against the process's current directory.
"""
from pathlib import Path
import os
import sys


def _location(variable, default):
    return Path(os.environ.get(variable, default)).expanduser().resolve()


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = _location('CIVOL_DATA_DIR', Path.cwd() / 'data')
WORK = _location('CIVOL_WORK_DIR', Path.cwd() / 'work')
CRSP_DAILY = _location('CIVOL_CRSP_DIR', DATA / 'crsp_daily')
FACTORS_DAILY = _location('CIVOL_FACTORS_FILE', DATA / 'factors_daily.csv')
MARKET_CALENDAR = _location('CIVOL_CALENDAR_FILE', DATA / 'experiment_ready' / 'market_calendar.parquet')
CACHE = _location('CIVOL_CACHE_DIR', WORK / 'cache')
GPU_PYTHON = _location('CIVOL_GPU_PYTHON', sys.executable)


def output_path(stage, *parts):
    """Return a generated artifact location without creating it on import."""
    return WORK.joinpath(stage, *parts)


def records_path(stage):
    """Override a record directory when replaying existing results."""
    return _location(f'CIVOL_{stage.upper()}_RECORDS', output_path(stage, 'records'))


RECORDS = records_path('application')
SIM_RECORDS = records_path('simulation')
