"""Input contracts exercised without network access or licensed observations."""
import io
import json
import zipfile

import pandas as pd
import pytest

from civol.data import download


def zipped(name, text):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(name, text)
    return output.getvalue()


def raw_frame():
    frame = pd.DataFrame({column: [None, None] for column in download.CRSP_COLUMNS})
    frame["permno"] = [1., 1.]
    frame["dlycaldt"] = pd.to_datetime(["1985-01-02", "1985-01-02"])
    frame["dlyprevdt"] = pd.to_datetime(["1984-12-31", "1984-12-31"])
    frame["dlyret"] = [0.1, None]
    frame["dlyprc"] = [-2., 2.]
    frame["dlycap"] = [100., 100.]
    return frame


def test_extract_preserves_duplicates_missing_values_negative_prices_and_units(tmp_path):
    class Database:
        calls = 0

        def raw_sql(self, sql, params, date_cols):
            self.calls += 1
            assert "FROM crsp.dsf_v2" in sql
            assert "ORDER BY dlycaldt, permno" in sql
            assert params == {"start_date": "1985-01-01", "end_date": "1985-12-31"}
            assert date_cols == ["dlycaldt", "dlyprevdt"]
            return raw_frame()

    db = Database()
    path = download.download_crsp_year(1985, db, tmp_path)
    result = pd.read_parquet(path)
    assert len(result) == 2
    assert result.permno.dtype == "int64"
    assert result.duplicated(["permno", "date"]).sum() == 1
    assert result.ret.iloc[0] == 0.1 and pd.isna(result.ret.iloc[1])
    assert result.prc.tolist() == [-2., 2.]
    assert result.mktcap.tolist() == [100., 100.]
    before = path.read_bytes()
    assert download.download_crsp_year(1985, db, tmp_path) == path
    assert db.calls == 1 and path.read_bytes() == before
    metadata = json.loads(download._sidecar(path).read_text())
    assert metadata["sha256"] == download.sha256(path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        download.download_crsp_year(1985, db, tmp_path)


def test_csv_factors_convert_percent_once_and_exclude_nondaily_rows():
    payload = zipped("daily.csv", "Header\n,Mkt-RF,SMB,HML,RF\n19841231,1.0,-2.0,0.5,0.02\n19850102,-1.0,2.0,-0.5,0.03\n1985,99,99,99,99\n")
    frame = download.parse_factor_csv(payload)
    assert list(frame.columns) == ["date", "Mkt-RF", "SMB", "HML", "RF"]
    assert len(frame) == 2
    assert frame["Mkt-RF"].tolist() == [0.01, -0.01]
    assert frame["RF"].tolist() == [0.0002, 0.0003]


def test_calendar_previous_session_is_computed_before_window_restriction():
    payload = zipped("daily.txt", "Header\n19841231 1.0 -2.0 0.5 0.02\n19850102 -1.0 2.0 -0.5 0.03\n19850103 0.5 0.1 0.2 0.04\n")
    full = download.parse_calendar_factors(payload)
    calendar = download.build_calendar(full, 1985, 1985)
    assert calendar.date.tolist() == list(pd.to_datetime(["1985-01-02", "1985-01-03"]))
    assert calendar.previous_market_date.tolist() == list(pd.to_datetime(["1984-12-31", "1985-01-02"]))
    assert calendar.month.tolist() == list(pd.to_datetime(["1985-01-31", "1985-01-31"]))


def test_existing_legacy_extract_is_never_modified(tmp_path):
    path = tmp_path / "crsp_daily_1985.parquet"
    download.transform_crsp(raw_frame()).to_parquet(path, index=False)
    before = path.read_bytes()
    assert download.download_crsp_year(1985, None, tmp_path) == path
    assert path.read_bytes() == before
    assert not download._sidecar(path).exists()


def test_missing_schema_fails_before_writing(tmp_path):
    class IncompleteDatabase:
        def raw_sql(self, *args, **kwargs):
            return raw_frame().drop(columns=["dlyretdurflg"])

    with pytest.raises(ValueError, match="dlyretdurflg"):
        download.download_crsp_year(1985, IncompleteDatabase(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_narrow_calendar_fails_clearly_and_preserves_existing_inputs(tmp_path, monkeypatch):
    factor_path = tmp_path / "factors.csv"
    calendar_path = tmp_path / "calendar.parquet"
    factor_dates = pd.to_datetime(["1984-12-28", "1984-12-31", "1985-01-02"])
    pd.DataFrame({"date": factor_dates}).to_csv(factor_path, index=False)
    calendar = pd.DataFrame({"date": [factor_dates[-1]], "previous_market_date": [factor_dates[-2]]})
    calendar.to_parquet(calendar_path, index=False)
    monkeypatch.setattr(download, "FACTORS_DAILY", factor_path)
    monkeypatch.setattr(download, "MARKET_CALENDAR", calendar_path)
    before = calendar_path.read_bytes()
    download.download_factors(1985, 1985)
    with pytest.raises(ValueError, match="Market calendar does not cover"):
        download.download_factors(1984, 1985)
    assert calendar_path.read_bytes() == before


def test_calendar_rejects_incorrect_previous_session():
    dates = pd.Series(pd.to_datetime(["1984-12-31", "1985-01-02", "1985-01-03"]))
    calendar = pd.DataFrame({"date": dates.iloc[1:].values,
                             "previous_market_date": pd.to_datetime(["1984-12-28", "1985-01-02"])})
    with pytest.raises(ValueError, match="previous-session dates disagree"):
        download.validate_calendar_coverage(calendar, dates, 1985, 1985)
