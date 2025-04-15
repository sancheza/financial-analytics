#!/usr/bin/env python3
import sys
import datetime
import contextlib
import io
from typing import Tuple
from get_bond_yield import fetch_auction_yield  # Uses your existing API module

VERSION = "1.04"

REINVESTMENT_RATE = 0.043

def print_help():
    print(f"""\
bond_return_calc.py — this script evaluates a secondary market bond's return and evaluates a comparable recent public auction

USAGE:
    bond_return_calc.py <coupon> <maturity_MM/DD/YYYY> <price> [--debug]

DESCRIPTION:
    Computes total compounded return and annualized return assuming {REINVESTMENT_RATE * 100:.1f}% coupon reinvestment on interest earned.
    Compares against the most recent Treasury auction yield with greater-or-equal maturity (e.g. 30Y).
    Input format matches E*TRADE bond listings.

OPTIONS:
    -h              Show this help message
    -v              Show version information
    --debug         Enable verbose API and filtering output
""")

def print_version():
    print(f"bond_return_calc.py version {VERSION}")

def future_value_of_coupons(coupon: float, rate: float, years: float) -> float:
    return coupon * (((1 + rate) ** years - 1) / rate)

def calculate_returns(coupon: float, maturity: datetime.date, price: float, reinvest_rate: float = REINVESTMENT_RATE):
    today = datetime.date.today()
    years = (maturity - today).days / 365.25
    fv_coupons = future_value_of_coupons(coupon, reinvest_rate, years)
    fv_total = fv_coupons + 100  # Face value
    total_return_pct = ((fv_total / price) - 1) * 100
    annual_return_pct = ((fv_total / price) ** (1 / years) - 1) * 100
    return fv_total, total_return_pct, annual_return_pct, years

def get_comparable_yield(maturity_date: datetime.date, debug: bool = False) -> Tuple[str, float]:
    years_remaining = (maturity_date - datetime.date.today()).days / 365.25

    if years_remaining > 20:
        duration_code = "30Y"
    elif years_remaining > 10:
        duration_code = "20Y"
    elif years_remaining > 7:
        duration_code = "10Y"
    elif years_remaining > 5:
        duration_code = "7Y"
    elif years_remaining > 3:
        duration_code = "5Y"
    elif years_remaining > 2:
        duration_code = "3Y"
    else:
        duration_code = "2Y"

    if debug:
        result = fetch_auction_yield(duration_code, datetime.datetime.today())
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            result = fetch_auction_yield(duration_code, datetime.datetime.today())

    if result and isinstance(result, tuple) and len(result) == 5:
        _, _, _, _, yield_value = result
        return duration_code, yield_value / 100  # Convert % to decimal
    else:
        raise RuntimeError(f"Failed to fetch benchmark yield for {duration_code} from Treasury API.")

def assign_grade(bond_return: float, benchmark: float) -> str:
    delta = bond_return - benchmark
    if delta >= 0.50: return "A+"
    elif delta >= 0.25: return "A"
    elif delta >= 0.10: return "A−"
    elif delta >= 0.00: return "B+"
    elif delta >= -0.09: return "B"
    elif delta >= -0.24: return "B−"
    elif delta >= -0.49: return "C"
    else: return "D"

def main():
    args = sys.argv[1:]

    if len(args) == 1 and args[0] == "-h":
        print_help(); sys.exit(0)
    elif len(args) == 1 and args[0] == "-v":
        print_version(); sys.exit(0)
    elif len(args) not in (3, 4):
        print_help(); sys.exit(1)

    debug = "--debug" in args
    args = [arg for arg in args if arg != "--debug"]

    try:
        coupon = float(args[0])
        maturity = datetime.datetime.strptime(args[1], "%m/%d/%Y").date()
        price = float(args[2])

        fv, total_return, annual_return, years = calculate_returns(coupon, maturity, price)

        duration_code, benchmark_yield = get_comparable_yield(maturity, debug=debug)
        benchmark_yield_pct = benchmark_yield * 100
        grade = assign_grade(annual_return, benchmark_yield_pct)

        # ANSI color codes
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        ORANGE = "\033[38;5;208m"
        GOLD = "\033[38;5;220m"
        RESET = "\033[0m"

        def color_value(val: str) -> str:
            return f"{GOLD}{val}{RESET}"

        def color_grade(g: str) -> str:
            return f"{GREEN}{g}{RESET}" if g in ("A+", "A", "A−", "B+", "B") else f"{RED}{g}{RESET}"

        print(f"Input: {coupon}% coupon, matures {maturity}, price ${price}")
        print(f"Annual Interest Payment: {color_value(f'${coupon:.2f}')} per year")
        print(f"Future Value (incl. reinvested coupons): {color_value(f'${fv:.2f}')}")
        print(f"Total Return: {color_value(f'{total_return:.2f}%')}")
        print(f"Annualized Return: {color_value(f'{annual_return:.2f}%')}")
        print(f"Benchmark Treasury Yield ({duration_code}): {color_value(f'{benchmark_yield_pct:.2f}%')}")
        print(f"Bond Grade: {color_grade(grade)}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
