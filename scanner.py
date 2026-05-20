from __future__ import annotations

import pandas as pd


BUY_POINT_COLUMNS = [
    "Close",
    "MACD",
    "Signal",
    "Momentum",
    "RSI",
    "MFI",
    "ConditionSummary",
]


def add_signal_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["macd_area"] = result["MACD"].apply(_macd_area)
    result["macd_flow"] = result.apply(_macd_flow, axis=1)

    prev_macd = result["MACD"].shift(1)
    prev_signal = result["Signal"].shift(1)
    result["macd_bullish_start"] = (
        (prev_signal > prev_macd) & (result["MACD"] > result["Signal"])
    ).fillna(False)
    result["macd_bearish_start"] = (
        (prev_macd > prev_signal) & (result["Signal"] > result["MACD"])
    ).fillna(False)

    result["momentum_ok"] = (result["Momentum"] > 0).fillna(False)
    result["rsi_ok"] = (result["RSI"] > 50).fillna(False)
    result["mfi_ok"] = (result["MFI"] > 50).fillna(False)
    result["all_three_ok"] = result["momentum_ok"] & result["rsi_ok"] & result["mfi_ok"]
    return result


def scan_buy_points(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = add_signal_columns(data)
    full["buy_point"] = False
    full["observation_active"] = False
    full["observation_start_date"] = pd.NaT

    buy_rows = []
    observing = False
    observation_start_date = None

    for date, row in full.iterrows():
        if observing and bool(row["macd_bearish_start"]):
            observing = False
            observation_start_date = None

        if (
            not observing
            and bool(row["macd_bullish_start"])
            and row["MACD"] < 0
        ):
            observing = True
            observation_start_date = date

        if observing:
            full.at[date, "observation_active"] = True
            full.at[date, "observation_start_date"] = observation_start_date

            if row["MACD"] > row["Signal"] and bool(row["all_three_ok"]):
                full.at[date, "buy_point"] = True
                buy_rows.append(_make_buy_point_row(date, row))
                observing = False
                observation_start_date = None

    buy_points = pd.DataFrame(buy_rows, columns=["Date", *BUY_POINT_COLUMNS])
    buy_points = buy_points.set_index("Date")
    return buy_points, full


def current_week_buy_point(
    buy_points: pd.DataFrame,
    scan_date: pd.Timestamp | str | None = None,
) -> pd.Series | None:
    week_start = _week_start(scan_date)
    if buy_points.empty or week_start not in buy_points.index:
        return None

    row = buy_points.loc[week_start]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def _week_start(scan_date: pd.Timestamp | str | None = None) -> pd.Timestamp:
    date = pd.Timestamp.today() if scan_date is None else pd.Timestamp(scan_date)
    normalized = date.normalize()
    return normalized - pd.Timedelta(days=normalized.weekday())


def _macd_area(macd: float) -> str:
    if pd.isna(macd):
        return ""
    if macd > 0:
        return "MACD 상승영역"
    if macd < 0:
        return "MACD 하락영역"
    return "기준선"


def _macd_flow(row: pd.Series) -> str:
    macd = row["MACD"]
    signal = row["Signal"]
    if pd.isna(macd) or pd.isna(signal):
        return ""
    if macd > signal:
        return "MACD 상승흐름"
    if signal > macd:
        return "MACD 하락흐름"
    return "교차 지점 / 중립"


def _make_buy_point_row(
    date: pd.Timestamp,
    row: pd.Series,
) -> dict[str, object]:
    return {
        "Date": date,
        "Close": row.get("Close"),
        "MACD": row["MACD"],
        "Signal": row["Signal"],
        "Momentum": row["Momentum"],
        "RSI": row["RSI"],
        "MFI": row["MFI"],
        "ConditionSummary": "MACD 상승흐름 유지 + Momentum > 0 + RSI > 50 + MFI > 50",
    }
