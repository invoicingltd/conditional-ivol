"""Targeted regression tests for the versioned formation-only rank safeguard."""
import json

import numpy as np
import pytest

from civol import safeguarded_measurement as sm
from civol.rank_rule.estimate_rank_rule import civol_at_rank


@pytest.fixture(autouse=True)
def serial_numerics():
    """Use the same deterministic numerical setting as legacy replay tests."""
    from numba import get_num_threads, set_num_threads
    from threadpoolctl import threadpool_limits
    previous = get_num_threads()
    set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        set_num_threads(previous)


def opposing_panel(T=120, N=80, seed=281):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, 1))
    loadings = np.tile([-1., 1.], N // 2)
    return .015 * (2.5 * f * loadings + rng.normal(size=(T, N)))


def test_opposing_common_direction_is_not_discarded_by_ew_rule():
    selected = sm.select_common_space(opposing_panel(), dict(T=120))
    diag = selected["diagnostics"]
    assert diag["ew_rank"] == 0
    assert diag["bai_ng_rank"] == 1
    assert selected["rank"] == 1
    assert diag["safeguard_changed_rank"] is True
    assert diag["selected_ew_ratio"] == diag["ew_curve"][1]


@pytest.mark.parametrize("curve,relative_attained", [
    ([3., 1.1, 1.4, 1.0], False),
    ([4., 1.8, 1.9, 1.5], False),
    ([4., 1.8, 1.6, 1.5], True),
])
def test_final_rank_checks_both_thresholds_on_nonmonotone_curve(monkeypatch, curve, relative_attained):
    monkeypatch.setattr(sm.rank_rules, "insample_curves", lambda *args, **kwargs:
                        dict(equal_weight=curve, kmax=3))
    monkeypatch.setattr(sm, "bai_ng_icp2", lambda *args, **kwargs: (2, [0., 0., 0.]))
    diag = sm.select_common_space(opposing_panel(T=30, N=24), dict(T=30))["diagnostics"]
    assert diag["ew_rank"] == 1 and diag["rank"] == 2
    assert diag["ew_selection_threshold"] == 1.25 * max(1., min(curve))
    assert diag["selected_ew_ratio"] == curve[2]
    assert diag["selected_ew_selection_threshold_satisfied"] is relative_attained
    assert diag["selected_ew_absolute_bound_satisfied"] is False


def test_at_selected_rank_outputs_equal_unchanged_legacy_kernel():
    x = opposing_panel(T=80, N=24)
    cfg = dict(T=80, H=3, pairs=64, batches=4, backend="numpy")
    prepared = sm.select_common_space(x, cfg)
    actual = sm.estimate_panel(x, base_seed=19, config=cfg)
    expected, model, basis = civol_at_rank(
        x, prepared["Z"], prepared["eig"], prepared["s"], prepared["mu"],
        prepared["rank"], 19 + 100 + prepared["rank"], prepared["config"])
    for name, values in expected.items():
        np.testing.assert_array_equal(actual["forecasts"][name], values)
    np.testing.assert_array_equal(actual["B_std"], basis)
    assert actual["summary"]["model"] == model
    assert actual["summary"]["version"] == sm.VERSION
    assert actual["summary"]["integration_seed"] == model["seed"]


def test_rank_safeguard_preserves_ew_when_it_already_dominates():
    assert sm.safeguarded_rank(9, 4) == 9
    assert sm.safeguarded_rank(0, 0) == 0
    assert sm.safeguarded_rank(2, 5) == 5
    with pytest.raises(ValueError):
        sm.safeguarded_rank(-1, 2)
    with pytest.raises(ValueError):
        sm.safeguarded_rank(2.5, 1)


def test_scientific_settings_cannot_silently_change_the_named_version():
    for changed in ({"bound": 1.5}, {"ew_kmax": 20}, {"bai_ng_kmax": 40},
                    {"seed_offset": 0}, {"factor_bound_multiple": 10.}):
        with pytest.raises(ValueError, match="would change"):
            sm.configuration(changed)
    with pytest.raises(ValueError, match="Unknown settings"):
        sm.configuration({"rank_rule": "icp2"})


def test_record_guard_refuses_existing_files_before_reading_data(tmp_path):
    class DatesOnlyCache:
        dates = np.array(["2020-01-02"], dtype="datetime64[D]")

    (tmp_path / "2020-01-02.npz").write_bytes(b"preserved partial record")
    with pytest.raises(FileExistsError, match="Refusing"):
        sm.run_date(DatesOnlyCache(), 0, out=tmp_path)
    assert (tmp_path / "2020-01-02.npz").read_bytes() == b"preserved partial record"


def test_complete_versioned_record_contains_ff3_measurement_comparison(tmp_path):
    class InventedCache:
        def __init__(self):
            T, N = 86, 24
            rng = np.random.default_rng(734)
            self.dates = np.datetime64("2020-01-01") + np.arange(T)
            self.ret = opposing_panel(T=T, N=N)
            self.ff = np.column_stack((rng.normal(0., .01, (T, 3)), np.zeros(T)))
            self.valid = np.ones((T, N), dtype=bool)
            self.elig = np.ones((T, N), dtype=bool)
            self.mktcap = np.ones((T, N))
            self.permnos = np.arange(N) + 1000
            self.permco = np.tile(np.arange(N) + 2000, (T, 1))

        def __len__(self):
            return len(self.dates)

    cache = InventedCache()
    result = sm.run_date(cache, 79, dict(T=80, H=6, pairs=64, batches=4), out=tmp_path)
    path = tmp_path / str(cache.dates[79])
    saved = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert arrays["q"].shape == (24,)
        assert np.all(arrays["q"] > 0.)
        assert arrays["scored"].sum() == 24
        assert set(("rv_ff3_fixed", "rv_broad", "rv_total")) <= set(arrays.files)
    assert saved["version"] == sm.VERSION == result["version"]
    assert saved["rank"] == max(saved["rank_diagnostics"]["ew_rank"],
                                saved["rank_diagnostics"]["bai_ng_rank"])
    assert saved["outcomes"]["scored_count"] == 24
    assert saved["source_hashes"]["safeguarded_measurement.py"]
