from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd

from data_provider import DataLoadError, load_weekly_data, normalize_ticker
from indicators import calculate_indicators
from market_cap_provider import MarketCapCompany, MarketCapLoadError, fetch_us_top_market_cap
from scanner import scan_buy_points


OUTPUT_DIR = Path("outputs")


class BuyPointApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("주봉 매수포인트 검증")
        self.geometry("1220x760")
        self.minsize(960, 600)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.ticker_var = tk.StringVar()
        self.status_var = tk.StringVar(value="티커를 입력해 주세요.")
        self.top100_status_var = tk.StringVar(value="목록을 불러오려면 버튼을 눌러 주세요.")

        self._build_layout()

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self, padding=14)
        main_frame.pack(fill="both", expand=True)
        main_frame.columnconfigure(0, minsize=360)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        left_panel = ttk.LabelFrame(main_frame, text="미국 시총 Top 100", padding=6)
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

        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(2, weight=1)
        right_panel.columnconfigure(0, weight=1)

        search_frame = ttk.Frame(right_panel)
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

        status_label = ttk.Label(right_panel, textvariable=self.status_var, padding=(0, 8, 0, 8))
        status_label.grid(row=1, column=0, sticky="ew")

        table_frame = ttk.LabelFrame(right_panel, text="매수포인트", padding=4)
        table_frame.grid(row=2, column=0, sticky="nsew")
        self.buy_tree = self._create_table(table_frame)

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
        tree.column("company", width=150, minwidth=120, stretch=True)
        tree.column("market_cap", width=84, minwidth=70, anchor="e", stretch=False)

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
        self.top100_tree.delete(*self.top100_tree.get_children())
        for company in companies:
            self.top100_tree.insert(
                "",
                "end",
                values=(company.rank, company.ticker, company.company, company.market_cap),
            )
        self.top100_status_var.set(f"{len(companies)}개 종목을 불러왔습니다. 행을 클릭하면 바로 검색합니다.")
        self.top100_button.configure(state="normal")

    def _show_top100_error(self, exc: Exception) -> None:
        if isinstance(exc, MarketCapLoadError):
            message = str(exc)
        else:
            message = f"미국 시총 Top 100 목록을 불러오지 못했습니다: {exc}"

        self.top100_tree.delete(*self.top100_tree.get_children())
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
        tree.column(column, width=_column_width(column), minwidth=80, stretch=True)

    for _, row in data.iterrows():
        values = [_format_value(row[column]) for column in columns]
        tree.insert("", "end", values=values)


def _column_width(column: str) -> int:
    if column in {"Date", "매수포인트날짜", "observation_start_date"}:
        return 150
    if column == "ConditionSummary":
        return 420
    if column in {"macd_area", "macd_flow"}:
        return 140
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
