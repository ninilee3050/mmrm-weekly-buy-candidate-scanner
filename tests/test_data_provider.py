from __future__ import annotations

import pandas as pd

from data_provider import (
    _chart_payload_to_dataframe,
    _drop_incomplete_current_week,
    _read_local_csv,
    _resample_daily_to_weekly,
)


def test_chart_payload_to_dataframe_normalizes_yahoo_weekly_rows() -> None:
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1704412800, 1705017600],
                    "indicators": {
                        "quote": [
                            {
                                "open": [100.0, 101.0],
                                "high": [105.0, 106.0],
                                "low": [99.0, 100.0],
                                "close": [104.0, 105.0],
                                "volume": [1000, 1100],
                            }
                        ],
                        "adjclose": [{"adjclose": [104.0, 105.0]}],
                    },
                }
            ],
            "error": None,
        }
    }

    result = _chart_payload_to_dataframe(payload)

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.iloc[0]["Open"] == 100.0
    assert result.iloc[1]["Close"] == 105.0


def test_drop_incomplete_current_week_removes_all_rows_in_current_week(monkeypatch) -> None:
    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def today(cls, tz=None):
            return cls("2026-05-20")

    monkeypatch.setattr(pd, "Timestamp", FixedTimestamp)
    data = pd.DataFrame(
        {"Close": [1.0, 2.0, 3.0, 4.0]},
        index=pd.to_datetime(["2026-05-08", "2026-05-15", "2026-05-18", "2026-05-19"]),
    )

    result = _drop_incomplete_current_week(data)

    assert result.index.strftime("%Y-%m-%d").tolist() == ["2026-05-08", "2026-05-15"]


def test_resample_daily_to_weekly_uses_monday_label_and_weekly_ohlcv() -> None:
    daily = pd.DataFrame(
        {
            "Open": [84.26, 85.00, 86.00, 87.00, 86.50],
            "High": [85.00, 87.83, 87.00, 87.20, 87.00],
            "Low": [84.08, 84.70, 85.80, 86.40, 86.20],
            "Close": [84.90, 86.20, 86.80, 86.50, 87.37],
            "Volume": [10, 20, 30, 40, 50],
            "Adj Close": [84.90, 86.20, 86.80, 86.50, 87.37],
        },
        index=pd.to_datetime(
            ["2025-08-18", "2025-08-19", "2025-08-20", "2025-08-21", "2025-08-22"]
        ),
    )

    result = _resample_daily_to_weekly(daily)

    assert result.index.strftime("%Y-%m-%d").tolist() == ["2025-08-18"]
    row = result.iloc[0]
    assert row["Open"] == 84.26
    assert row["High"] == 87.83
    assert row["Low"] == 84.08
    assert row["Close"] == 87.37
    assert row["Volume"] == 150


def test_resample_daily_to_weekly_can_keep_current_week(monkeypatch) -> None:
    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def today(cls, tz=None):
            return cls("2026-05-20")

    monkeypatch.setattr(pd, "Timestamp", FixedTimestamp)
    daily = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.0, 12.0],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2026-05-18", "2026-05-19"]),
    )

    result = _resample_daily_to_weekly(daily, include_current_week=True)

    assert result.index.strftime("%Y-%m-%d").tolist() == ["2026-05-18"]
    assert result.iloc[0]["Close"] == 12.0


def test_read_local_csv_can_keep_or_drop_current_week(tmp_path, monkeypatch) -> None:
    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def today(cls, tz=None):
            return cls("2026-05-20")

    monkeypatch.setattr(pd, "Timestamp", FixedTimestamp)
    csv_path = tmp_path / "TEST.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2026-05-11,1,2,1,2,100",
                "2026-05-18,2,3,2,3,200",
            ]
        ),
        encoding="utf-8",
    )

    dropped = _read_local_csv(csv_path)
    kept = _read_local_csv(csv_path, include_current_week=True)

    assert dropped.index.strftime("%Y-%m-%d").tolist() == ["2026-05-11"]
    assert kept.index.strftime("%Y-%m-%d").tolist() == ["2026-05-11", "2026-05-18"]
