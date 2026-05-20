from __future__ import annotations

import pandas as pd

from scanner import add_signal_columns, current_week_buy_point, scan_buy_points


def make_indicator_frame(rows: list[dict[str, float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-05", periods=len(rows), freq="W-FRI")
    data = pd.DataFrame(rows, index=index)
    if "Close" not in data.columns:
        data["Close"] = 100.0
    return data


def test_macd_bullish_start_is_detected_from_strict_flow_change() -> None:
    data = make_indicator_frame(
        [
            {"MACD": -2.0, "Signal": -1.0, "Momentum": -1.0, "RSI": 40.0, "MFI": 40.0},
            {"MACD": -0.8, "Signal": -1.2, "Momentum": -1.0, "RSI": 40.0, "MFI": 40.0},
        ]
    )

    result = add_signal_columns(data)

    assert result.iloc[0]["macd_flow"] == "MACD 하락흐름"
    assert bool(result.iloc[1]["macd_bullish_start"]) is True
    assert result.iloc[1]["macd_flow"] == "MACD 상승흐름"


def test_momentum_rsi_mfi_conditions_use_strict_thresholds() -> None:
    data = make_indicator_frame(
        [
            {"MACD": -1.0, "Signal": -2.0, "Momentum": 0.0, "RSI": 50.0, "MFI": 50.0},
            {"MACD": -1.0, "Signal": -2.0, "Momentum": 0.01, "RSI": 50.01, "MFI": 50.01},
        ]
    )

    result = add_signal_columns(data)

    assert bool(result.iloc[0]["momentum_ok"]) is False
    assert bool(result.iloc[0]["rsi_ok"]) is False
    assert bool(result.iloc[0]["mfi_ok"]) is False
    assert bool(result.iloc[0]["all_three_ok"]) is False
    assert bool(result.iloc[1]["momentum_ok"]) is True
    assert bool(result.iloc[1]["rsi_ok"]) is True
    assert bool(result.iloc[1]["mfi_ok"]) is True
    assert bool(result.iloc[1]["all_three_ok"]) is True


def test_buy_point_is_recorded_after_lower_area_bullish_start_and_three_conditions() -> None:
    data = make_indicator_frame(
        [
            {"MACD": -2.0, "Signal": -1.0, "Momentum": -3.0, "RSI": 40.0, "MFI": 40.0},
            {"MACD": -0.8, "Signal": -1.2, "Momentum": -1.0, "RSI": 48.0, "MFI": 49.0},
            {"MACD": -0.4, "Signal": -0.9, "Momentum": 0.2, "RSI": 51.0, "MFI": 52.0},
            {"MACD": 0.2, "Signal": -0.3, "Momentum": 1.2, "RSI": 60.0, "MFI": 61.0},
        ]
    )

    buy_points, full = scan_buy_points(data)

    assert len(buy_points) == 1
    assert buy_points.index[0] == data.index[2]
    assert "ObservationStartDate" not in buy_points.columns
    assert bool(full.loc[data.index[2], "buy_point"]) is True
    assert bool(full.loc[data.index[3], "buy_point"]) is False


def test_observation_resets_when_bearish_start_appears_before_conditions() -> None:
    data = make_indicator_frame(
        [
            {"MACD": -2.0, "Signal": -1.0, "Momentum": -3.0, "RSI": 40.0, "MFI": 40.0},
            {"MACD": -0.8, "Signal": -1.2, "Momentum": -1.0, "RSI": 48.0, "MFI": 49.0},
            {"MACD": -1.2, "Signal": -0.9, "Momentum": -0.5, "RSI": 49.0, "MFI": 49.0},
            {"MACD": -1.1, "Signal": -0.8, "Momentum": 0.5, "RSI": 55.0, "MFI": 56.0},
            {"MACD": -0.6, "Signal": -1.0, "Momentum": 0.8, "RSI": 58.0, "MFI": 59.0},
        ]
    )

    buy_points, full = scan_buy_points(data)

    assert bool(full.loc[data.index[2], "macd_bearish_start"]) is True
    assert bool(full.loc[data.index[3], "buy_point"]) is False
    assert len(buy_points) == 1
    assert buy_points.index[0] == data.index[4]
    assert "ObservationStartDate" not in buy_points.columns


def test_current_week_buy_point_returns_only_scan_week_candidate() -> None:
    buy_points = pd.DataFrame(
        {
            "Close": [100.0, 110.0],
            "MACD": [-1.0, -0.5],
            "Signal": [-1.5, -0.8],
            "Momentum": [1.0, 2.0],
            "RSI": [55.0, 60.0],
            "MFI": [56.0, 61.0],
            "ConditionSummary": ["old", "current"],
        },
        index=pd.to_datetime(["2026-05-11", "2026-05-18"]),
    )

    result = current_week_buy_point(buy_points, scan_date="2026-05-20")

    assert result is not None
    assert result.name == pd.Timestamp("2026-05-18")
    assert result["ConditionSummary"] == "current"


def test_current_week_buy_point_ignores_previous_week_candidate() -> None:
    buy_points = pd.DataFrame(
        {
            "Close": [100.0],
            "MACD": [-1.0],
            "Signal": [-1.5],
            "Momentum": [1.0],
            "RSI": [55.0],
            "MFI": [56.0],
            "ConditionSummary": ["old"],
        },
        index=pd.to_datetime(["2026-05-11"]),
    )

    result = current_week_buy_point(buy_points, scan_date="2026-05-20")

    assert result is None
