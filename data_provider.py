from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


DATA_DIR = Path("data")
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


class DataLoadError(RuntimeError):
    """Raised when Yahoo Finance cannot provide ticker data."""


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("티커를 입력해 주세요.")
    return cleaned


def load_weekly_data(
    ticker: str,
    data_dir: Path | str = DATA_DIR,
    include_current_week: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"{ticker}.csv"
    if csv_path.exists() and not force_refresh:
        try:
            cached = _read_local_csv(csv_path, include_current_week=include_current_week)
            if _is_monday_labeled_weekly(cached):
                return cached
        except Exception:
            pass

    data = _download_weekly_from_yahoo(ticker, include_current_week=include_current_week)
    if data.empty:
        raise DataLoadError(f"{ticker} 데이터가 비어 있습니다. 티커를 다시 확인해 주세요.")

    data.to_csv(csv_path, index_label="Date", encoding="utf-8-sig")
    return data


def _read_local_csv(path: Path, include_current_week: bool = False) -> pd.DataFrame:
    raw = pd.read_csv(path)
    data = _normalize_ohlcv_dataframe(raw)
    if include_current_week:
        return data
    return _drop_incomplete_current_week(data)


def _download_weekly_from_yahoo(ticker: str, include_current_week: bool = False) -> pd.DataFrame:
    errors = []

    try:
        return _download_daily_then_resample_from_yahoo(
            ticker,
            include_current_week=include_current_week,
        )
    except Exception as exc:  # pragma: no cover - depends on network.
        errors.append(f"Yahoo daily chart API: {exc}")

    try:
        return _download_from_yahoo_chart(
            ticker,
            interval="1wk",
            include_current_week=include_current_week,
        )
    except Exception as exc:  # pragma: no cover - depends on network.
        errors.append(f"Yahoo weekly chart API: {exc}")

    try:
        return _download_with_yfinance(ticker, include_current_week=include_current_week)
    except Exception as exc:  # pragma: no cover - depends on optional package/network.
        errors.append(f"yfinance: {exc}")

    details = " / ".join(errors)
    raise DataLoadError(
        f"{ticker} 데이터를 야후파이낸스에서 불러오지 못했습니다. "
        f"인터넷 연결과 티커를 확인해 주세요. 상세: {details}"
    )


def _download_with_yfinance(ticker: str, include_current_week: bool = False) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise DataLoadError("yfinance가 설치되어 있지 않습니다.") from exc

    raw = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    data = _normalize_ohlcv_dataframe(raw.reset_index())
    if data.empty:
        raise DataLoadError("yfinance에서 빈 데이터가 반환되었습니다.")
    return _resample_daily_to_weekly(data, include_current_week=include_current_week)


def _download_daily_then_resample_from_yahoo(
    ticker: str,
    include_current_week: bool = False,
) -> pd.DataFrame:
    daily = _download_from_yahoo_chart(
        ticker,
        interval="1d",
        include_current_week=include_current_week,
    )
    return _resample_daily_to_weekly(daily, include_current_week=include_current_week)


def _download_from_yahoo_chart(
    ticker: str,
    interval: str,
    include_current_week: bool = False,
) -> pd.DataFrame:
    period2 = int(time.time())
    query = (
        f"?period1=0&period2={period2}"
        f"&interval={interval}&events=history&includeAdjustedClose=true"
    )
    url = YAHOO_CHART_URL.format(ticker=quote(ticker, safe="")) + query
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise DataLoadError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise DataLoadError(str(exc.reason)) from exc

    data = _chart_payload_to_dataframe(payload)
    if data.empty:
        raise DataLoadError("Yahoo chart API에서 빈 데이터가 반환되었습니다.")
    if include_current_week:
        return data
    return _drop_incomplete_current_week(data)


def _chart_payload_to_dataframe(payload: dict) -> pd.DataFrame:
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "알 수 없는 오류"
        raise DataLoadError(str(description))

    results = chart.get("result") or []
    if not results:
        raise DataLoadError("응답에 가격 데이터가 없습니다.")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose_data = (result.get("indicators", {}).get("adjclose") or [{}])[0]

    rows = {
        "Date": [
            datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
            for timestamp in timestamps
        ],
        "Open": quote_data.get("open", []),
        "High": quote_data.get("high", []),
        "Low": quote_data.get("low", []),
        "Close": quote_data.get("close", []),
        "Volume": quote_data.get("volume", []),
    }

    adjclose = adjclose_data.get("adjclose")
    if adjclose is not None:
        rows["Adj Close"] = adjclose

    return _normalize_ohlcv_dataframe(pd.DataFrame(rows))


def _normalize_ohlcv_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    if data.empty and len(data.columns) == 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    data.columns = [str(column).strip() for column in data.columns]

    lower_to_original = {column.lower(): column for column in data.columns}
    rename_map = {}
    for expected in ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        original = lower_to_original.get(expected.lower())
        if original:
            rename_map[original] = expected
    data = data.rename(columns=rename_map)

    if "Date" not in data.columns:
        first_column = data.columns[0]
        data = data.rename(columns={first_column: "Date"})

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise DataLoadError(f"가격 데이터에 필요한 컬럼이 없습니다: {missing_text}")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "Adj Close" in data.columns:
        data["Adj Close"] = pd.to_numeric(data["Adj Close"], errors="coerce")

    data = data.dropna(subset=REQUIRED_COLUMNS)
    return data


def _drop_incomplete_current_week(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    today = pd.Timestamp.today().normalize()
    current_week_start = _monday_week_start(pd.DatetimeIndex([today]))[0]
    row_week_starts = _monday_week_start(data.index)
    return data[row_week_starts < current_week_start]


def _resample_daily_to_weekly(
    daily: pd.DataFrame,
    include_current_week: bool = False,
) -> pd.DataFrame:
    if daily.empty:
        return daily

    data = daily.sort_index().copy()
    data["_WeekStart"] = _monday_week_start(data.index)

    aggregations = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    if "Adj Close" in data.columns:
        aggregations["Adj Close"] = "last"

    weekly = data.groupby("_WeekStart").agg(aggregations)
    weekly.index = pd.to_datetime(weekly.index)
    weekly.index.name = "Date"
    weekly = weekly.dropna(subset=REQUIRED_COLUMNS)
    if include_current_week:
        return weekly
    return _drop_incomplete_current_week(weekly)


def _monday_week_start(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized = pd.DatetimeIndex(index).normalize()
    return normalized - pd.to_timedelta(normalized.weekday, unit="D")


def _is_monday_labeled_weekly(data: pd.DataFrame) -> bool:
    if data.empty:
        return False
    return bool((pd.DatetimeIndex(data.index).weekday == 0).all())
