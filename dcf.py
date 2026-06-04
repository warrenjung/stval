#!/usr/bin/env python3
"""
FCF-based DCF valuation using Yahoo Finance data.

Requirements:
  pip install yfinance numpy

Example:
  python dcf.py VRSN
  python dcf.py MSFT --market-return 0.10 --terminal-growth 0.03
  python dcf.py VRSN --analyst-source none
  python dcf.py HSY --beta-method blume
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings as py_warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np

py_warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL.*",
)

import yfinance as yf


DEFAULT_MARKET_RETURN = 0.10
DEFAULT_TERMINAL_GROWTH = 0.03
DEFAULT_DECAY_FACTOR = 0.5
DEFAULT_CLAMP_K = 1.0
DEFAULT_ANALYST_WEIGHT = 0.10
DEFAULT_ANALYST_SOURCE = "revenue"
DEFAULT_BETA_METHOD = "raw"
DEFAULT_TAX_RATE = 0.21
DEFAULT_ADJUSTMENT = 0.0


OCF_ROWS = (
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
)
CAPEX_ROWS = (
    "Capital Expenditure",
    "Capital Expenditures",
    "Capital Expenditure Reported",
)
INTEREST_ROWS = (
    "Interest Expense",
    "Interest Expense Non Operating",
    "Interest Expense, Non Operating",
)
TAX_ROWS = ("Tax Provision", "Income Tax Expense")
PRETAX_ROWS = ("Pretax Income", "Income Before Tax", "Earnings Before Tax")
DEBT_ROWS = ("Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation")
SHARES_ROWS = ("Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding")


@dataclass
class HistoricalFCF:
    label: str
    ocf: float
    capex_raw: float
    capex_spend: float
    adjustment: float
    fcf: float
    growth: float | None = None


@dataclass
class Projection:
    year: int
    growth: float
    fcf: float
    pv: float


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def finite_number(value: object) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def first_info_number(info: dict, keys: Iterable[str]) -> float | None:
    for key in keys:
        value = finite_number(info.get(key))
        if value is not None:
            return value
    return None


def first_statement_value(statement, row_names: Iterable[str], column=None) -> float | None:
    if statement is None or getattr(statement, "empty", True):
        return None

    for row in row_names:
        if row not in statement.index:
            continue
        series = statement.loc[row]
        if column is not None and column in series.index:
            value = finite_number(series[column])
            if value is not None:
                return value
        for item in series:
            value = finite_number(item)
            if value is not None:
                return value
    return None


def available_statement_columns(statement) -> list:
    if statement is None or getattr(statement, "empty", True):
        return []

    date_columns = [column for column in statement.columns if hasattr(column, "strftime")]
    if date_columns:
        return sorted(date_columns)
    return sorted(statement.columns, key=str)


def column_label(column) -> str:
    if hasattr(column, "strftime"):
        return column.strftime("%Y-%m-%d")
    return str(column)


def get_statement_pair(statement, column) -> tuple[float | None, float | None]:
    ocf = first_statement_value(statement, OCF_ROWS, column)
    capex = first_statement_value(statement, CAPEX_ROWS, column)
    return ocf, capex


def build_historical_fcf(ticker: yf.Ticker, adjustment: float, warnings: list[str]) -> list[HistoricalFCF]:
    # STEP 1 - Historical FCF: 4 most recent fiscal years plus TTM.
    annual_cashflow = ticker.cashflow
    quarterly_cashflow = ticker.quarterly_cashflow

    historical: list[HistoricalFCF] = []
    annual_columns = available_statement_columns(annual_cashflow)
    recent_annual_columns = annual_columns[-4:]

    for column in recent_annual_columns:
        ocf, capex_raw = get_statement_pair(annual_cashflow, column)
        if ocf is None or capex_raw is None:
            warn(warnings, f"Skipping fiscal year {column_label(column)} because OCF or CapEx is missing.")
            continue
        capex_spend = abs(capex_raw)
        fcf = ocf - capex_spend + adjustment
        historical.append(
            HistoricalFCF(
                label=column_label(column),
                ocf=ocf,
                capex_raw=capex_raw,
                capex_spend=capex_spend,
                adjustment=adjustment,
                fcf=fcf,
            )
        )

    quarterly_columns = available_statement_columns(quarterly_cashflow)
    recent_quarters = quarterly_columns[-4:]
    if len(recent_quarters) == 4:
        ttm_ocf = 0.0
        ttm_capex_raw = 0.0
        missing_quarters = []
        for column in recent_quarters:
            ocf, capex_raw = get_statement_pair(quarterly_cashflow, column)
            if ocf is None or capex_raw is None:
                missing_quarters.append(column_label(column))
                continue
            ttm_ocf += ocf
            ttm_capex_raw += capex_raw

        if missing_quarters:
            warn(warnings, f"TTM excludes missing quarter data: {', '.join(missing_quarters)}.")
        elif ttm_ocf != 0 or ttm_capex_raw != 0:
            capex_spend = abs(ttm_capex_raw)
            ttm_fcf = ttm_ocf - capex_spend + adjustment
            historical.append(
                HistoricalFCF(
                    label="TTM",
                    ocf=ttm_ocf,
                    capex_raw=ttm_capex_raw,
                    capex_spend=capex_spend,
                    adjustment=adjustment,
                    fcf=ttm_fcf,
                )
            )
            if historical[-2:-1] and abs(historical[-2].fcf - ttm_fcf) <= max(abs(ttm_fcf), 1.0) * 0.005:
                warn(warnings, "TTM FCF is very close to the latest fiscal-year FCF; it may duplicate annual data.")
    else:
        warn(warnings, "Could not compute TTM because fewer than four quarterly cash-flow periods were available.")

    if len(historical) < 5:
        warn(warnings, f"Only {len(historical)} usable FCF observations found; growth estimates may be weak.")

    for index in range(1, len(historical)):
        previous = historical[index - 1].fcf
        current = historical[index].fcf
        if previous == 0:
            warn(warnings, f"Growth for {historical[index].label} is unavailable because prior FCF is zero.")
            continue
        historical[index].growth = current / previous - 1.0
        if previous < 0 or current < 0:
            warn(warnings, f"Growth for {historical[index].label} crosses or uses negative FCF; interpret cautiously.")

    return historical


def regression_growth(fcf_values: list[float], warnings: list[str]) -> tuple[float, float]:
    # STEP 2a - LOGEST-equivalent exponential trend on ln(FCF).
    if len(fcf_values) < 2:
        warn(warnings, "Regression growth unavailable: fewer than two FCF values.")
        return 0.0, 0.0
    if any(value <= 0 for value in fcf_values):
        warn(warnings, "Regression growth unavailable: LOGEST-style fit requires all FCF values to be positive.")
        return 0.0, 0.0

    periods = np.arange(1, len(fcf_values) + 1, dtype=float)
    y = np.log(np.array(fcf_values, dtype=float))
    slope, intercept = np.polyfit(periods, y, 1)
    y_hat = slope * periods + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return math.exp(float(slope)) - 1.0, r2


def time_weighted_growth(
    raw_growth_rates: list[float],
    decay_factor: float,
    clamp_k: float,
    warnings: list[str],
) -> tuple[float, list[float]]:
    # STEP 2b - Clamp outliers and weight recent growth more heavily.
    rates = [rate for rate in raw_growth_rates if rate is not None and math.isfinite(rate)]
    if not rates:
        warn(warnings, "Time-weighted average growth unavailable: no usable growth observations.")
        return 0.0, []

    mean = float(np.mean(rates))
    sample_std = float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0
    lower = mean - clamp_k * sample_std
    upper = mean + clamp_k * sample_std
    cleaned = [float(np.median([rate, lower, upper])) for rate in rates]

    newest_first = list(reversed(cleaned))
    weights = [decay_factor**years_ago for years_ago in range(len(newest_first))]
    weighted_avg = sum(weight * rate for weight, rate in zip(weights, newest_first)) / sum(weights)
    return weighted_avg, cleaned


def analyst_growth_rate(
    info: dict,
    override: float | None,
    source: str,
    warnings: list[str],
) -> tuple[float | None, str]:
    # STEP 2c - Optional analyst growth input.
    if override is not None:
        return override, "CLI override"

    if source == "none":
        warn(warnings, "Analyst growth source set to none; analyst blend weight set to 0.")
        return None, "disabled"

    if source == "revenue":
        keys = ("revenueGrowth",)
    else:
        keys = ("earningsGrowth", "earningsQuarterlyGrowth")

    for key in keys:
        value = finite_number(info.get(key))
        if value is not None:
            return value, f"Yahoo info['{key}']"

    warn(warnings, f"No {source} analyst growth estimate found; analyst blend weight set to 0.")
    return None, "unavailable"


def project_fcf(
    starting_fcf: float,
    blended_growth: float,
    terminal_growth: float,
    wacc: float,
) -> list[Projection]:
    # STEP 3 and STEP 5 - Ten-year FCF projection and present value.
    projections: list[Projection] = []
    previous_fcf = starting_fcf
    previous_growth = blended_growth

    for year in range(1, 11):
        if year <= 5:
            growth = blended_growth
        else:
            growth = previous_growth + (terminal_growth - previous_growth) / 5.0
        fcf = previous_fcf * (1.0 + growth)
        pv = fcf / ((1.0 + wacc) ** year)
        projections.append(Projection(year=year, growth=growth, fcf=fcf, pv=pv))
        previous_fcf = fcf
        previous_growth = growth

    return projections


def effective_tax_rate(ticker: yf.Ticker, warnings: list[str]) -> float:
    income_stmt = ticker.income_stmt
    tax = first_statement_value(income_stmt, TAX_ROWS)
    pretax = first_statement_value(income_stmt, PRETAX_ROWS)
    if tax is None or pretax in (None, 0):
        warn(warnings, "Effective tax rate unavailable; falling back to 21%.")
        return DEFAULT_TAX_RATE

    rate = tax / pretax
    if not math.isfinite(rate) or rate < 0 or rate > 0.6:
        warn(warnings, f"Effective tax rate looked unusual ({format_percent(rate)}); falling back to 21%.")
        return DEFAULT_TAX_RATE
    return rate


def interest_expense(ticker: yf.Ticker, warnings: list[str]) -> float:
    income_stmt = ticker.income_stmt
    value = first_statement_value(income_stmt, INTEREST_ROWS)
    if value is None:
        warn(warnings, "Interest expense unavailable; using 0.")
        return 0.0
    return abs(value)


def total_debt(ticker: yf.Ticker, info: dict, warnings: list[str]) -> float:
    info_debt = first_info_number(info, ("totalDebt",))
    if info_debt is not None:
        return max(0.0, info_debt)

    balance_sheet = ticker.balance_sheet
    statement_debt = first_statement_value(balance_sheet, DEBT_ROWS)
    if statement_debt is not None:
        warn(warnings, "Total debt unavailable from .info; using balance-sheet fallback.")
        return max(0.0, statement_debt)

    warn(warnings, "Total debt unavailable; using 0.")
    return 0.0


def shares_outstanding(ticker: yf.Ticker, info: dict, warnings: list[str]) -> float | None:
    shares = first_info_number(info, ("sharesOutstanding", "impliedSharesOutstanding"))
    if shares is not None and shares > 0:
        return shares

    balance_sheet = ticker.balance_sheet
    shares = first_statement_value(balance_sheet, SHARES_ROWS)
    if shares is not None and shares > 0:
        warn(warnings, "Shares outstanding unavailable from .info; using balance-sheet fallback.")
        return shares

    warn(warnings, "Shares outstanding unavailable; intrinsic price cannot be computed.")
    return None


def market_cap(info: dict, warnings: list[str]) -> float | None:
    value = first_info_number(info, ("marketCap",))
    if value is None or value <= 0:
        warn(warnings, "Market cap unavailable; WACC cannot be computed.")
        return None
    return value


def beta(info: dict, warnings: list[str]) -> float:
    value = first_info_number(info, ("beta",))
    if value is None:
        warn(warnings, "Beta unavailable; using 1.0.")
        return 1.0
    return value


def apply_beta_method(raw_beta: float, method: str) -> float:
    if method == "blume":
        return 0.67 * raw_beta + 0.33
    return raw_beta


def risk_free_rate(warnings: list[str]) -> float:
    try:
        tnx = yf.Ticker("^TNX")
        history = tnx.history(period="5d")
        if history.empty or "Close" not in history:
            raise ValueError("no ^TNX close price")
        rate = float(history["Close"].dropna().iloc[-1]) / 100.0
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("invalid ^TNX close price")
        return rate
    except Exception as exc:  # yfinance can fail in several network/data-provider ways.
        warn(warnings, f"Risk-free rate unavailable from ^TNX ({exc}); using 4.0%.")
        return 0.04


def compute_wacc(
    market_cap_value: float,
    debt: float,
    beta_value: float,
    risk_free: float,
    market_return: float,
    interest: float,
    tax_rate: float,
    warnings: list[str],
) -> tuple[float, float, float]:
    # STEP 4 - WACC.
    cost_of_equity = risk_free + beta_value * (market_return - risk_free)
    if debt > 0:
        after_tax_cost_of_debt = (interest / debt) * (1.0 - tax_rate)
    else:
        after_tax_cost_of_debt = 0.0
        warn(warnings, "Total debt is zero; after-tax cost of debt set to 0.")

    equity = market_cap_value
    total_capital = equity + debt
    if total_capital <= 0:
        raise ValueError("Market cap plus debt must be greater than zero.")

    # Sheet discount-rate formula: WACC = (E/V)*CAPM cost of equity + (D/V)*after-tax cost of debt.
    wacc = (equity / total_capital) * cost_of_equity + (debt / total_capital) * after_tax_cost_of_debt
    return cost_of_equity, after_tax_cost_of_debt, wacc


def terminal_value(year_10_fcf: float, terminal_growth: float, wacc: float) -> float:
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({format_percent(wacc)}) must be greater than terminal growth "
            f"({format_percent(terminal_growth)}) for terminal value."
        )
    return year_10_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)


def format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    return f"{value:,.2f}"


def format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.2f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:,.2f}%"


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_outputs(
    ticker_symbol: str,
    historical: list[HistoricalFCF],
    cleaned_growth: list[float],
    reg_growth: float,
    r2: float,
    time_weighted_avg: float,
    analyst_rate: float | None,
    analyst_source: str,
    analyst_weight: float,
    blended_growth: float,
    projections: list[Projection],
    market_cap_value: float,
    debt: float,
    raw_beta: float,
    beta_value: float,
    beta_method: str,
    tax_rate: float,
    interest: float,
    risk_free: float,
    market_return: float,
    cost_of_equity: float,
    after_tax_cost_of_debt: float,
    wacc: float,
    pv_explicit: float,
    pv_terminal: float,
    enterprise_value: float,
    shares: float,
    intrinsic_price: float,
    warnings: list[str],
) -> None:
    print(f"\nFCF DCF Valuation: {ticker_symbol.upper()}")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    historical_rows = []
    for item in historical:
        historical_rows.append(
            [
                item.label,
                format_money(item.ocf),
                format_money(item.capex_raw),
                format_money(item.capex_spend),
                format_money(item.adjustment),
                format_money(item.fcf),
                format_percent(item.growth),
            ]
        )
    print_table(
        "Step 1 - Historical FCF & Growth",
        ["Period", "OCF", "CapEx Raw", "CapEx Spend", "Adjustment", "FCF", "YoY Growth"],
        historical_rows,
    )

    growth_rows = [
        ["Regression Growth", format_percent(reg_growth)],
        ["Regression R^2", format_number(r2)],
        ["Time-Weighted Average", format_percent(time_weighted_avg)],
        ["Analyst Rate", f"{format_percent(analyst_rate)} ({analyst_source})"],
        ["Analyst Weight", format_percent(analyst_weight)],
        ["Blended Growth", format_percent(blended_growth)],
    ]
    if cleaned_growth:
        growth_rows.append(["Cleaned Growth Rates", ", ".join(format_percent(rate) for rate in cleaned_growth)])
    print_table("Step 2 - Blended Near-Term Growth Rate", ["Metric", "Value"], growth_rows)

    projection_rows = [
        [str(item.year), format_percent(item.growth), format_money(item.fcf), format_money(item.pv)]
        for item in projections
    ]
    print_table("Step 3/5 - 10-Year FCF Projection", ["Year", "Growth", "FCF", "Discounted PV"], projection_rows)

    wacc_rows = [
        ["Market Cap (E)", format_money(market_cap_value)],
        ["Total Debt (D)", format_money(debt)],
        ["Raw Beta", format_number(raw_beta)],
        ["Beta Method", beta_method],
        ["Beta Used", format_number(beta_value)],
        ["Effective Tax Rate", format_percent(tax_rate)],
        ["Interest Expense", format_money(interest)],
        ["Risk-Free Rate", format_percent(risk_free)],
        ["Market Return", format_percent(market_return)],
        ["Cost of Equity", format_percent(cost_of_equity)],
        ["After-Tax Cost of Debt", format_percent(after_tax_cost_of_debt)],
        ["WACC", format_percent(wacc)],
    ]
    print_table("Step 4 - WACC / Discount Rate", ["Metric", "Value"], wacc_rows)

    valuation_rows = [
        ["PV Explicit (Sum)", format_money(pv_explicit)],
        ["PV Terminal (Terminal)", format_money(pv_terminal)],
        ["Enterprise Value (Total)", format_money(enterprise_value)],
        ["Total Debt", format_money(debt)],
        ["Shares Outstanding", format_money(shares)],
        ["Intrinsic Price (Price)", f"${intrinsic_price:,.2f}"],
    ]
    print_table("Step 5 - Valuation", ["Metric", "Value"], valuation_rows)

    if warnings:
        print("\nWarnings")
        print("--------")
        for message in warnings:
            print(f"- {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute an FCF-based DCF intrinsic price from Yahoo Finance data.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. VRSN")
    parser.add_argument("--market-return", type=float, default=DEFAULT_MARKET_RETURN, help="Expected market return.")
    parser.add_argument(
        "--analyst-rate",
        type=float,
        default=None,
        help="Override analyst growth rate as a decimal, e.g. 0.08 for 8%%.",
    )
    parser.add_argument(
        "--analyst-source",
        choices=("revenue", "earnings", "none"),
        default=DEFAULT_ANALYST_SOURCE,
        help="Yahoo growth proxy to blend when --analyst-rate is not supplied.",
    )
    parser.add_argument("--terminal-growth", type=float, default=DEFAULT_TERMINAL_GROWTH, help="Terminal growth rate.")
    parser.add_argument("--adjustment", type=float, default=DEFAULT_ADJUSTMENT, help="Manual annual/TTM FCF adjustment.")
    parser.add_argument("--decay-factor", type=float, default=DEFAULT_DECAY_FACTOR, help="Time-weight decay factor.")
    parser.add_argument("--clamp-k", type=float, default=DEFAULT_CLAMP_K, help="Growth clamp sample-stdev multiplier.")
    parser.add_argument("--analyst-weight", type=float, default=DEFAULT_ANALYST_WEIGHT, help="Analyst blend weight.")
    parser.add_argument(
        "--beta-method",
        choices=("raw", "blume"),
        default=DEFAULT_BETA_METHOD,
        help="Use raw Yahoo beta or Blume-adjusted beta: 0.67 * raw_beta + 0.33.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not -1.0 < args.market_return < 1.0:
        raise ValueError("--market-return should be a decimal rate, e.g. 0.10.")
    if not -1.0 < args.terminal_growth < 1.0:
        raise ValueError("--terminal-growth should be a decimal rate, e.g. 0.03.")
    if args.analyst_rate is not None and not -1.0 < args.analyst_rate < 1.0:
        raise ValueError("--analyst-rate should be a decimal rate, e.g. 0.08.")
    if not 0.0 <= args.decay_factor <= 1.0:
        raise ValueError("--decay-factor must be between 0 and 1.")
    if args.clamp_k < 0:
        raise ValueError("--clamp-k must be non-negative.")
    if not 0.0 <= args.analyst_weight <= 1.0:
        raise ValueError("--analyst-weight must be between 0 and 1.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    warnings: list[str] = []
    ticker_symbol = args.ticker.upper()
    ticker = yf.Ticker(ticker_symbol)

    try:
        info = ticker.info or {}
    except Exception as exc:
        info = {}
        warn(warnings, f"Could not load ticker .info ({exc}); using statement fallbacks where possible.")

    historical = build_historical_fcf(ticker, args.adjustment, warnings)
    if len(historical) < 2:
        print("Not enough cash-flow history to compute a DCF.", file=sys.stderr)
        for message in warnings:
            print(f"Warning: {message}", file=sys.stderr)
        return 1

    fcf_values = [item.fcf for item in historical]
    raw_growth = [item.growth for item in historical[1:] if item.growth is not None]
    reg_growth, r2 = regression_growth(fcf_values, warnings)
    time_weighted_avg, cleaned_growth = time_weighted_growth(raw_growth, args.decay_factor, args.clamp_k, warnings)

    analyst_rate, analyst_source = analyst_growth_rate(info, args.analyst_rate, args.analyst_source, warnings)
    effective_analyst_weight = args.analyst_weight if analyst_rate is not None else 0.0
    core_growth = r2 * reg_growth + (1.0 - r2) * time_weighted_avg
    blended_growth = effective_analyst_weight * (analyst_rate or 0.0) + (1.0 - effective_analyst_weight) * core_growth

    market_cap_value = market_cap(info, warnings)
    shares = shares_outstanding(ticker, info, warnings)
    if market_cap_value is None or shares is None:
        print("Missing required market cap or share count; cannot compute valuation.", file=sys.stderr)
        for message in warnings:
            print(f"Warning: {message}", file=sys.stderr)
        return 1

    debt = total_debt(ticker, info, warnings)
    raw_beta = beta(info, warnings)
    beta_value = apply_beta_method(raw_beta, args.beta_method)
    tax_rate = effective_tax_rate(ticker, warnings)
    interest = interest_expense(ticker, warnings)
    risk_free = risk_free_rate(warnings)

    try:
        cost_of_equity, after_tax_cost_of_debt, wacc = compute_wacc(
            market_cap_value=market_cap_value,
            debt=debt,
            beta_value=beta_value,
            risk_free=risk_free,
            market_return=args.market_return,
            interest=interest,
            tax_rate=tax_rate,
            warnings=warnings,
        )
        projections = project_fcf(historical[-1].fcf, blended_growth, args.terminal_growth, wacc)
        pv_explicit = sum(item.pv for item in projections)
        tv = terminal_value(projections[-1].fcf, args.terminal_growth, wacc)
        pv_terminal = tv / ((1.0 + wacc) ** 10)
        enterprise_value = pv_explicit + pv_terminal
        intrinsic_price = (enterprise_value - debt) / shares
    except ValueError as exc:
        print(f"Valuation error: {exc}", file=sys.stderr)
        for message in warnings:
            print(f"Warning: {message}", file=sys.stderr)
        return 1

    print_outputs(
        ticker_symbol=ticker_symbol,
        historical=historical,
        cleaned_growth=cleaned_growth,
        reg_growth=reg_growth,
        r2=r2,
        time_weighted_avg=time_weighted_avg,
        analyst_rate=analyst_rate,
        analyst_source=analyst_source,
        analyst_weight=effective_analyst_weight,
        blended_growth=blended_growth,
        projections=projections,
        market_cap_value=market_cap_value,
        debt=debt,
        raw_beta=raw_beta,
        beta_value=beta_value,
        beta_method=args.beta_method,
        tax_rate=tax_rate,
        interest=interest,
        risk_free=risk_free,
        market_return=args.market_return,
        cost_of_equity=cost_of_equity,
        after_tax_cost_of_debt=after_tax_cost_of_debt,
        wacc=wacc,
        pv_explicit=pv_explicit,
        pv_terminal=pv_terminal,
        enterprise_value=enterprise_value,
        shares=shares,
        intrinsic_price=intrinsic_price,
        warnings=warnings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
