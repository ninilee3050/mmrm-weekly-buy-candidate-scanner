from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from data_provider import DataLoadError, load_weekly_data, normalize_ticker
from indicators import calculate_indicators
from market_cap_provider import MarketCapCompany, MarketCapLoadError, fetch_us_top_market_cap
from scanner import current_week_buy_point, scan_buy_points


OUTPUT_DIR = Path("outputs")
DOWNLOADS_DIR = Path.home() / "Downloads"
BUY_DISPLAY_COLUMNS = [
    "매수포인트날짜",
    "Close",
    "MACD",
    "Signal",
    "Momentum",
    "RSI",
    "MFI",
    "ConditionSummary",
]
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


class BuyPointApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("주봉 매수포인트 검증")
        self.geometry("2520x820")
        self.minsize(1800, 680)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.ticker_var = tk.StringVar()
        self.status_var = tk.StringVar(value="티커를 입력해 주세요.")
        self.top100_status_var = tk.StringVar(value="목록을 불러오려면 버튼을 눌러 주세요.")
        self.scan_status_var = tk.StringVar(value="스캐너를 실행하려면 버튼을 눌러 주세요.")
        self.top100_companies: list[MarketCapCompany] = []
        self.latest_scan_candidates = pd.DataFrame(columns=SCAN_RESULT_COLUMNS)
        self.latest_scan_failures = pd.DataFrame(columns=SCAN_FAILURE_COLUMNS)
        self.latest_scan_date: pd.Timestamp | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self, padding=14)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, minsize=430)
        main_frame.columnconfigure(1, weight=1, minsize=930)
        main_frame.columnconfigure(2, weight=1, minsize=1110)
        main_frame.rowconfigure(0, weight=1)

        left_panel = ttk.LabelFrame(main_frame, text="미국 시총 Top 100", padding=6)
        left_panel.configure(width=430)
        left_panel.grid_propagate(False)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)

        self.top100_button = ttk.Button(
            left_panel,
            text="Top 100 불러오기",
            command=self.load_top100,
        )
        self.top100_button.grid(row=0, column=0, sticky="ew")

        top100_status = ttk.Label(
            left_panel,
            textvariable=self.top100_status_var,
            wraplength=330,
            padding=(0, 6, 0, 6),
        )
        top100_status.grid(row=1, column=0, sticky="ew")

        self.top100_tree = self._create_top100_table(left_panel)
        self.top100_tree.bind("<<TreeviewSelect>>", self._on_top100_select)

        center_panel = ttk.Frame(main_frame)
        center_panel.configure(width=930)
        center_panel.grid_propagate(False)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        center_panel.rowconfigure(2, weight=1)
        center_panel.columnconfigure(0, weight=1)

        search_frame = ttk.Frame(center_panel)
        search_frame.grid(row=0, column=0, sticky="ew")
        search_frame.columnconfigure(0, weight=1)

        self.search_entry = ttk.Entry(
            search_frame,
            textvariable=self.ticker_var,
            font=("Segoe UI", 16),
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", ipady=6)
        self.search_entry.bind("<Return>", lambda _event: self.run_search())
        self.search_entry.focus_set()

        self.search_button = ttk.Button(
            search_frame,
            text="검색",
            command=self.run_search,
        )
        self.search_button.grid(row=0, column=1, padx=(8, 0), ipady=4)

        status_label = ttk.Label(center_panel, textvariable=self.status_var, padding=(0, 8, 0, 8))
        status_label.grid(row=1, column=0, sticky="ew")

        table_frame = ttk.LabelFrame(center_panel, text="매수포인트", padding=4)
        table_frame.grid(row=2, column=0, sticky="nsew")
        self.buy_tree = self._create_table(table_frame)
        populate_table(self.buy_tree, pd.DataFrame(columns=BUY_DISPLAY_COLUMNS))

        scanner_panel = ttk.LabelFrame(main_frame, text="이번주 매수후보 스캐너", padding=6)
        scanner_panel.configure(width=1110)
        scanner_panel.grid_propagate(False)
        scanner_panel.grid(row=0, column=2, sticky="nsew")
        scanner_panel.rowconfigure(2, weight=1)
        scanner_panel.columnconfigure(0, weight=1)

        scan_button_frame = ttk.Frame(scanner_panel)
        scan_button_frame.grid(row=0, column=0, sticky="ew")
        scan_button_frame.columnconfigure(0, weight=1)
        scan_button_frame.columnconfigure(1, weight=1)

        self.scan_button = ttk.Button(
            scan_button_frame,
            text="Top 100 스캔",
            command=self.run_top100_scan,
        )
        self.scan_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.scan_save_button = ttk.Button(
            scan_button_frame,
            text="스캔 저장하기",
            command=self.save_latest_scan,
            state="disabled",
        )
        self.scan_save_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        scan_status = ttk.Label(
            scanner_panel,
            textvariable=self.scan_status_var,
            wraplength=400,
            padding=(0, 6, 0, 6),
        )
        scan_status.grid(row=1, column=0, sticky="ew")

        scan_table_frame = ttk.Frame(scanner_panel)
        scan_table_frame.grid(row=2, column=0, sticky="nsew")
        scan_table_frame.rowconfigure(0, weight=1)
        scan_table_frame.columnconfigure(0, weight=1)
        self.scan_tree = self._create_table(scan_table_frame)
        populate_table(self.scan_tree, pd.DataFrame(columns=SCAN_RESULT_COLUMNS))
        self.scan_tree.bind("<<TreeviewSelect>>", self._on_scan_candidate_select)

    def _create_table(self, parent: tk.Widget) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)

        tree = ttk.Treeview(frame, show="headings")
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _create_top100_table(self, parent: tk.Widget) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.grid(row=2, column=0, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = ["rank", "ticker", "company", "market_cap"]
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        tree.heading("rank", text="순위")
        tree.heading("ticker", text="티커")
        tree.heading("company", text="회사명")
        tree.heading("market_cap", text="시가총액")
        tree.column("rank", width=52, minwidth=45, anchor="center", stretch=False)
        tree.column("ticker", width=76, minwidth=60, anchor="center", stretch=False)
        tree.column("company", width=170, minwidth=150, stretch=False)
        tree.column("market_cap", width=100, minwidth=95, anchor="e", stretch=False)

        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        return tree

    def load_top100(self) -> None:
        self.top100_button.configure(state="disabled")
        self.top100_status_var.set("미국 시총 Top 100 목록을 불러오는 중입니다...")
        self.top100_tree.delete(*self.top100_tree.get_children())

        worker = threading.Thread(target=self._top100_worker, daemon=True)
        worker.start()

    def _top100_worker(self) -> None:
        try:
            companies = fetch_us_top_market_cap(limit=100)
        except Exception as exc:
            self.after(0, self._show_top100_error, exc)
            return

        self.after(0, self._show_top100_result, companies)

    def _show_top100_result(self, companies: list[MarketCapCompany]) -> None:
        self._populate_top100_table(companies)
        self.top100_status_var.set(f"{len(companies)}개 종목을 불러왔습니다. 행을 클릭하면 바로 검색합니다.")
        self.top100_button.configure(state="normal")

    def _populate_top100_table(self, companies: list[MarketCapCompany]) -> None:
        self.top100_companies = list(companies)
        self.top100_tree.delete(*self.top100_tree.get_children())
        for company in companies:
            self.top100_tree.insert(
                "",
                "end",
                values=(company.rank, company.ticker, company.company, company.market_cap),
            )

    def _show_top100_error(self, exc: Exception) -> None:
        if isinstance(exc, MarketCapLoadError):
            message = str(exc)
        else:
            message = f"미국 시총 Top 100 목록을 불러오지 못했습니다: {exc}"

        self.top100_tree.delete(*self.top100_tree.get_children())
        self.top100_companies = []
        self.top100_status_var.set("목록을 불러오지 못했습니다.")
        self.top100_button.configure(state="normal")
        messagebox.showerror("Top 100 조회 실패", message)

    def _on_top100_select(self, _event) -> None:
        selected = self.top100_tree.selection()
        if not selected:
            return
        ticker = self.top100_tree.set(selected[0], "ticker")
        if not ticker:
            return
        self.ticker_var.set(ticker)
        self.run_search()

    def run_top100_scan(self) -> None:
        if str(self.scan_button.cget("state")) == "disabled":
            return

        self.scan_button.configure(state="disabled")
        self.scan_save_button.configure(state="disabled")
        self.top100_button.configure(state="disabled")
        self.scan_status_var.set("Top 100 스캔을 준비하는 중입니다...")
        self.scan_tree.delete(*self.scan_tree.get_children())
        self.latest_scan_candidates = pd.DataFrame(columns=SCAN_RESULT_COLUMNS)
        self.latest_scan_failures = pd.DataFrame(columns=SCAN_FAILURE_COLUMNS)
        self.latest_scan_date = None

        companies = list(self.top100_companies)
        worker = threading.Thread(
            target=self._top100_scan_worker,
            args=(companies,),
            daemon=True,
        )
        worker.start()

    def _top100_scan_worker(self, companies: list[MarketCapCompany]) -> None:
        try:
            if not companies:
                self.after(0, self.scan_status_var.set, "Top 100 목록을 먼저 불러오는 중입니다...")
                companies = fetch_us_top_market_cap(limit=100)
                self.after(0, self._show_top100_loaded_by_scan, companies)

            scan_date = pd.Timestamp.today().normalize()
            candidates, failures = self._scan_companies(
                companies,
                scan_date,
                progress_label="스캔 중",
            )

            if failures:
                time.sleep(2)
                retry_companies = [failure["company"] for failure in failures]
                retry_candidates, retry_failures = self._scan_companies(
                    retry_companies,
                    scan_date,
                    progress_label="실패 종목 재시도 중",
                )
                candidates.extend(retry_candidates)
                failures = retry_failures

            candidates_df = pd.DataFrame(candidates, columns=SCAN_RESULT_COLUMNS)
            failures_df = pd.DataFrame(
                [_failure_row(failure["company"], failure["error"]) for failure in failures],
                columns=SCAN_FAILURE_COLUMNS,
            )
        except Exception as exc:
            self.after(0, self._show_scan_error, exc)
            return

        self.after(
            0,
            self._show_scan_result,
            candidates_df,
            failures_df,
            scan_date,
        )

    def _show_top100_loaded_by_scan(self, companies: list[MarketCapCompany]) -> None:
        self._populate_top100_table(companies)
        self.top100_status_var.set(f"{len(companies)}개 종목을 불러왔습니다. 이 목록을 기준으로 스캔합니다.")

    def _scan_companies(
        self,
        companies: list[MarketCapCompany],
        scan_date: pd.Timestamp,
        progress_label: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        candidates: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        total = len(companies)

        for index, company in enumerate(companies, start=1):
            self.after(
                0,
                self.scan_status_var.set,
                f"{progress_label}... {index}/{total} {company.ticker}",
            )
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
                    candidates.append(_scan_candidate_row(company, candidate, scan_date))
            except Exception as exc:
                failures.append({"company": company, "error": str(exc)})

        return candidates, failures

    def _show_scan_result(
        self,
        candidates: pd.DataFrame,
        failures: pd.DataFrame,
        scan_date: pd.Timestamp,
    ) -> None:
        self.latest_scan_candidates = candidates.copy()
        self.latest_scan_failures = failures.copy()
        self.latest_scan_date = scan_date
        populate_table(self.scan_tree, candidates)

        failed_tickers = ", ".join(failures["티커"].tolist()[:8]) if not failures.empty else ""
        failed_suffix = f": {failed_tickers}" if failed_tickers else ""
        if len(failures) > 8:
            failed_suffix += "..."

        self.scan_status_var.set(
            f"스캔 완료: 이번주 매수후보 {len(candidates)}개 / "
            f"최종 실패 {len(failures)}개{failed_suffix}. "
            f"필요하면 스캔 저장하기를 눌러 CSV로 저장하세요."
        )
        self.scan_button.configure(state="normal")
        self.scan_save_button.configure(state="normal")
        self.top100_button.configure(state="normal")

    def _show_scan_error(self, exc: Exception) -> None:
        if isinstance(exc, MarketCapLoadError):
            message = str(exc)
        else:
            message = f"Top 100 스캔 중 오류가 발생했습니다: {exc}"

        self.scan_status_var.set(message)
        self.scan_button.configure(state="normal")
        self.scan_save_button.configure(state="disabled")
        self.top100_button.configure(state="normal")
        messagebox.showerror("Top 100 스캔 실패", message)

    def save_latest_scan(self) -> None:
        if self.latest_scan_date is None:
            messagebox.showinfo("저장할 스캔 없음", "먼저 Top 100 스캔을 실행해 주세요.")
            return

        candidate_path, failure_path = save_top100_scan_outputs(
            self.latest_scan_candidates,
            self.latest_scan_failures,
            self.latest_scan_date,
        )
        self.scan_status_var.set(
            f"스캔 결과 저장 완료: {candidate_path} / {failure_path}"
        )

    def _on_scan_candidate_select(self, _event) -> None:
        selected = self.scan_tree.selection()
        if not selected:
            return
        ticker = self.scan_tree.set(selected[0], "티커")
        if not ticker:
            return
        self.ticker_var.set(ticker)
        self.run_search()

    def run_search(self) -> None:
        if str(self.search_button.cget("state")) == "disabled":
            return

        try:
            ticker = normalize_ticker(self.ticker_var.get())
        except ValueError as exc:
            messagebox.showinfo("입력 필요", str(exc))
            return

        self.search_button.configure(state="disabled")
        self.status_var.set(f"{ticker} 주봉 데이터를 불러오는 중입니다...")

        worker = threading.Thread(target=self._search_worker, args=(ticker,), daemon=True)
        worker.start()

    def _search_worker(self, ticker: str) -> None:
        try:
            raw_data = load_weekly_data(ticker)
            calculated = calculate_indicators(raw_data)
            buy_points, full_table = scan_buy_points(calculated)
            buy_path, full_path = save_outputs(ticker, buy_points, full_table)
        except Exception as exc:
            self.after(0, self._show_error, ticker, exc)
            return

        self.after(
            0,
            self._show_result,
            ticker,
            buy_points,
            buy_path,
        )

    def _show_result(
        self,
        ticker: str,
        buy_points: pd.DataFrame,
        buy_path: Path,
    ) -> None:
        populate_table(self.buy_tree, table_for_display(buy_points))

        count = len(buy_points)
        self.status_var.set(
            f"{ticker}: 매수포인트 {count}개를 찾았습니다. "
            f"저장: {buy_path}"
        )
        self.search_button.configure(state="normal")

    def _show_error(self, ticker: str, exc: Exception) -> None:
        if isinstance(exc, DataLoadError):
            message = str(exc)
        else:
            message = f"{ticker} 처리 중 오류가 발생했습니다: {exc}"

        self.status_var.set(message)
        self.search_button.configure(state="normal")
        messagebox.showerror("오류", message)


def save_outputs(
    ticker: str,
    buy_points: pd.DataFrame,
    full_table: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buy_path = output_dir / f"{ticker}_buy_points.csv"
    full_path = output_dir / f"{ticker}_full_table.csv"

    buy_points.to_csv(buy_path, index_label="매수포인트날짜", encoding="utf-8-sig")
    full_table.to_csv(full_path, index_label="Date", encoding="utf-8-sig")
    return buy_path, full_path


def save_top100_scan_outputs(
    candidates: pd.DataFrame,
    failures: pd.DataFrame,
    scan_date: pd.Timestamp,
    output_dir: Path | str = DOWNLOADS_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_text = scan_date.strftime("%Y-%m-%d")
    candidate_path = output_dir / f"top100_scan_candidates_{date_text}.csv"
    failure_path = output_dir / f"top100_scan_failures_{date_text}.csv"

    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    failures.to_csv(failure_path, index=False, encoding="utf-8-sig")
    return candidate_path, failure_path


def _scan_candidate_row(
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


def table_for_display(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data.reset_index().rename(columns={"Date": "매수포인트날짜"})
    display = data.reset_index()
    if "Date" not in display.columns:
        display = display.rename(columns={display.columns[0]: "Date"})
    display = display.rename(columns={"Date": "매수포인트날짜"})
    return display


def populate_table(tree: ttk.Treeview, data: pd.DataFrame) -> None:
    tree.delete(*tree.get_children())
    columns = list(data.columns)
    tree["columns"] = columns

    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=_column_width(column), minwidth=60, stretch=False)

    for _, row in data.iterrows():
        values = [_format_value(row[column]) for column in columns]
        tree.insert("", "end", values=values)


def _column_width(column: str) -> int:
    if column in {"Date", "매수포인트날짜", "observation_start_date", "주봉시작일", "스캔일"}:
        return 110
    if column in {"회사명", "오류"}:
        return 160
    if column == "ConditionSummary":
        return 330
    if column in {"macd_area", "macd_flow"}:
        return 140
    if column in {"순위", "티커"}:
        return 70
    if column == "시가총액":
        return 95
    if column in {"Close", "MACD", "Signal", "RSI", "MFI"}:
        return 75
    if column == "Momentum":
        return 85
    return 110


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


if __name__ == "__main__":
    app = BuyPointApp()
    app.mainloop()
