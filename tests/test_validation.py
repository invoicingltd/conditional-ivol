"""Saved-result comparison must reject scientific differences by default."""
import json

import numpy as np
import pandas as pd
import pytest

from civol.validation import compare_results


@pytest.fixture
def result_directories(tmp_path):
    expected, actual = tmp_path / "expected", tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    return expected, actual


def write_result(folder, extension, value=1.0, label="civol"):
    path = folder / ("table." + extension)
    if extension == "json":
        path.write_text(json.dumps({"model": label, "loss": value, "missing": float("nan")}),
                        encoding="utf-8")
    else:
        frame = pd.DataFrame({"model": [label, "benchmark"], "loss": [value, np.nan]})
        if extension == "csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)


@pytest.mark.parametrize("extension", ["csv", "json", "parquet"])
def test_identical_results_and_missing_values_match(result_directories, extension):
    expected, actual = result_directories
    for folder in result_directories:
        write_result(folder, extension)
    assert compare_results(expected, actual) == {
        "matched": 1, "baseline_tables": 1, "replay_tables": 1, "errors": [],
        "rtol": 0, "atol": 0}


@pytest.mark.parametrize("extension", ["csv", "json", "parquet"])
def test_numerical_changes_fail_exactly_and_pass_only_with_requested_tolerance(result_directories, extension):
    expected, actual = result_directories
    write_result(expected, extension, value=1.0)
    write_result(actual, extension, value=1.0 + 1e-10)
    exact = compare_results(expected, actual)
    assert exact["matched"] == 0 and len(exact["errors"]) == 1
    tolerant = compare_results(expected, actual, rtol=1e-9, atol=0.0)
    assert tolerant["matched"] == 1 and tolerant["errors"] == []


@pytest.mark.parametrize("extension", ["csv", "json", "parquet"])
def test_tolerance_never_hides_changed_model_labels(result_directories, extension):
    expected, actual = result_directories
    write_result(expected, extension, label="civol")
    write_result(actual, extension, label="ff3")
    result = compare_results(expected, actual, rtol=1.0, atol=1.0)
    assert result["matched"] == 0 and len(result["errors"]) == 1


def test_csv_round_trip_precision_preserves_distinct_float_values(result_directories):
    expected, actual = result_directories
    (expected / "table.csv").write_text("value\n0.30000000000000004\n", encoding="utf-8")
    (actual / "table.csv").write_text("value\n0.3\n", encoding="utf-8")
    result = compare_results(expected, actual)
    assert result["matched"] == 0 and len(result["errors"]) == 1


@pytest.mark.parametrize("change", ["rows", "columns"])
def test_table_order_is_part_of_the_result(result_directories, change):
    expected, actual = result_directories
    frame = pd.DataFrame({"model": ["civol", "ff3"], "loss": [1.2, 1.4]})
    frame.to_csv(expected / "table.csv", index=False)
    reordered = frame.iloc[::-1] if change == "rows" else frame[["loss", "model"]]
    reordered.to_csv(actual / "table.csv", index=False)
    result = compare_results(expected, actual, rtol=1.0, atol=1.0)
    assert result["matched"] == 0 and len(result["errors"]) == 1


def test_missing_and_unexpected_tables_are_reported(result_directories):
    expected, actual = result_directories
    for folder in result_directories:
        write_result(folder, "csv")
    (expected / "missing.json").write_text("{}", encoding="utf-8")
    (actual / "extra.json").write_text("{}", encoding="utf-8")
    result = compare_results(expected, actual)
    assert result["matched"] == 1
    assert result["baseline_tables"] == result["replay_tables"] == 2
    assert set(result["errors"]) == {"Missing replay: missing.json", "Unexpected replay: extra.json"}


def test_run_manifest_metadata_is_ignored_but_csv_manifest_is_a_table(result_directories):
    expected, actual = result_directories
    for folder in result_directories:
        write_result(folder, "csv")
    (expected / "RUN_MANIFEST.json").write_text('{"source": "original"}', encoding="utf-8")
    (actual / "RUN_MANIFEST_20261022.json").write_text('{"source": "refactored"}', encoding="utf-8")
    assert compare_results(expected, actual)["errors"] == []
    (expected / "RUN_MANIFEST.csv").write_text("value\n1\n", encoding="utf-8")
    assert compare_results(expected, actual)["errors"] == ["Missing replay: RUN_MANIFEST.csv"]


def test_elapsed_seconds_excluded_but_stored_estimation_time_preserved(result_directories):
    expected, actual = result_directories
    baseline = {"seconds": 10.0, "seconds_per_date_mean": 2.5, "loss": 0.75}
    replay = dict(baseline, seconds=20.0)
    (expected / "SUMMARY.json").write_text(json.dumps(baseline), encoding="utf-8")
    (actual / "SUMMARY.json").write_text(json.dumps(replay), encoding="utf-8")
    assert compare_results(expected, actual)["errors"] == []
    replay["seconds_per_date_mean"] = 3.0
    (actual / "SUMMARY.json").write_text(json.dumps(replay), encoding="utf-8")
    result = compare_results(expected, actual)
    assert result["matched"] == 0 and len(result["errors"]) == 1


def test_empty_baseline_cannot_report_success(result_directories):
    result = compare_results(*result_directories)
    assert result["matched"] == 0
    assert result["errors"] == ["No baseline result tables found"]
