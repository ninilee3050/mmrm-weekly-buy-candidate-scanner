from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from data_provider import load_weekly_data
from indicators import calculate_indicators
from market_cap_provider import MarketCapCompany, fetch_us_top_market_cap
from scanner import current_week_buy_point, scan_buy_points


OUTPUT_DIR = Path("outputs")
TOP100_LIMIT = 100
RETRY_DELAY_SECONDS = 2

SCAN_RESULT_COLUMNS = [
    "순위",
    "티커",
    "회사명",
    "시가총액",
    "주봉시작일",
    "스캔일",
    "Close",
    "MACD",
    "Signal",
    "Momentum",
    "RSI",
    "MFI",
]
SCAN_FAILURE_COLUMNS = ["순위", "티커", "회사명", "시가총액", "오류"]


def main() -> int:
    try:
        scan_date = pd.Timestamp.today().normalize()
        print("미국 시가총액 Top 100 목록을 불러옵니다...")
        companies = fetch_us_top_market_cap(limit=TOP100_LIMIT)

        candidates, failures = scan_companies(companies, scan_date, progress_label="스캔 중")
        if failures:
            print(f"실패한 {len(failures)}개 종목을 한 번 더 시도합니다...")
            time.sleep(RETRY_DELAY_SECONDS)
            retry_companies = [failure["company"] for failure in failures]
            retry_candidates, retry_failures = scan_companies(
                retry_companies,
                scan_date,
                progress_label="재시도 중",
            )
            candidates.extend(retry_candidates)
            failures = retry_failures

        candidates_df = pd.DataFrame(candidates, columns=SCAN_RESULT_COLUMNS)
        failures_df = pd.DataFrame(
            [_failure_row(failure["company"], failure["error"]) for failure in failures],
            columns=SCAN_FAILURE_COLUMNS,
        )
        candidates_df = candidates_df.sort_values("순위").reset_index(drop=True)
        failures_df = failures_df.sort_values("순위").reset_index(drop=True)

        candidate_path, failure_path = save_scan_outputs(candidates_df, failures_df, scan_date)
    except Exception as exc:
        print(f"스캔 실패: {exc}", file=sys.stderr)
        return 1

    print("")
    print(f"스캔 완료: 이번주 매수후보 {len(candidates_df)}개 / 최종 실패 {len(failures_df)}개")
    print(f"매수후보 저장: {candidate_path}")
    print(f"실패목록 저장: {failure_path}")
    if not candidates_df.empty:
        print("")
        print("이번주 매수후보:")
        for _, row in candidates_df.iterrows():
            print(f"- {row['순위']}위 {row['티커']} {row['회사명']}")
    return 0


def scan_companies(
    companies: list[MarketCapCompany],
    scan_date: pd.Timestamp,
    progress_label: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    total = len(companies)

    for index, company in enumerate(companies, start=1):
        print(f"{progress_label}... {index}/{total} {company.ticker}")
        try:
            raw_data = load_weekly_data(
                company.ticker,
                include_current_week=True,
                force_refresh=True,
            )
            calculated = calculate_indicators(raw_data)
            buy_points, _full_table = scan_buy_points(calculated)
            candidate = current_week_buy_point(buy_points, scan_date)
            if candidate is not None:
                candidates.append(_candidate_row(company, candidate, scan_date))
        except Exception as exc:
            failures.append({"company": company, "error": str(exc)})

    return candidates, failures


def save_scan_outputs(
    candidates: pd.DataFrame,
    failures: pd.DataFrame,
    scan_date: pd.Timestamp,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_text = scan_date.strftime("%Y-%m-%d")
    candidate_path = output_dir / f"weekly_scan_candidates_{date_text}.csv"
    failure_path = output_dir / f"weekly_scan_failures_{date_text}.csv"

    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    failures.to_csv(failure_path, index=False, encoding="utf-8-sig")
    return candidate_path, failure_path


def _candidate_row(
    company: MarketCapCompany,
    candidate: pd.Series,
    scan_date: pd.Timestamp,
) -> dict[str, object]:
    return {
        "순위": company.rank,
        "티커": company.ticker,
        "회사명": company.company,
        "시가총액": company.market_cap,
        "주봉시작일": candidate.name,
        "스캔일": scan_date,
        "Close": candidate.get("Close"),
        "MACD": candidate.get("MACD"),
        "Signal": candidate.get("Signal"),
        "Momentum": candidate.get("Momentum"),
        "RSI": candidate.get("RSI"),
        "MFI": candidate.get("MFI"),
    }


def _failure_row(company: MarketCapCompany, error: str) -> dict[str, object]:
    return {
        "순위": company.rank,
        "티커": company.ticker,
        "회사명": company.company,
        "시가총액": company.market_cap,
        "오류": error,
    }


if __name__ == "__main__":
    raise SystemExit(main())
