#!/usr/bin/env python3
import sys
import datetime
import contextlib
import io
from typing import Tuple, List, Optional
from scipy.optimize import newton
from dateutil.relativedelta import relativedelta
from get_bond_yield import fetch_auction_yield  # Uses your existing API module

VERSION = "1.10" # Updated version
SETTLEMENT_LAG_DAYS = 1 # Use T+1 for Treasuries


def print_help():
    print(f"""\
bond_return_calc.py — this script evaluates a secondary market bond's return and evaluates a comparable recent public auction

USAGE:
    bond_return_calc.py <coupon> <maturity_MM/DD/YYYY> <price> [--debug]

DESCRIPTION:
    Computes Yield to Maturity (YTM) using standard Bond Equivalent Yield (BEY) conventions (T+1 settlement, Actual/Actual day count, semi-annual compounding).
    Compares against the most recent Treasury auction yield with greater-or-equal maturity (e.g. 30Y).
    Also shows a simplified total return assuming coupons are held until maturity (not reinvested).
    Input format matches E*TRADE bond listings, e.g. bond_return_calc.py 4.625 01/31/2025 99.75

OPTIONS:
    -h              Show this help message
    -v              Show version information
    --debug         Enable verbose API and filtering output
""")


def print_version():
    print(f"bond_return_calc.py version {VERSION}")


def calculate_returns(coupon_rate: float, maturity: datetime.date, price: float, face_value: float = 100):
    """
    Calculates simplified total/annual return assuming coupons are held (0% reinvestment).
    Note: This calculation uses simple year fractions and doesn't match bond math precisely.
    Returns: Tuple[float, float, float, float, int] -> fv_total, total_return_pct, annual_return_pct, years, num_future_coupons
    """
    today = datetime.date.today()
    # Assuming weekends/holidays are handled by simply adding days.
    settlement_date = today + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)
    years = (maturity - today).days / 365.25 # Simple approximation for annualization

    # Calculate total value of coupons received from settlement until maturity
    coupon_amt_semi_annual = (coupon_rate / 100.0 * face_value) / 2.0
    all_coupon_dates = get_coupon_dates(maturity, settlement_date) # Use existing helper
    future_coupon_dates = [d for d in all_coupon_dates if d > settlement_date]
    num_future_coupons = len(future_coupon_dates) # Number of coupons counted
    total_coupons_value = num_future_coupons * coupon_amt_semi_annual

    fv_total = face_value + total_coupons_value  # Final value = Principal + Sum of coupons received

    total_return_pct = ((fv_total / price) - 1) * 100
    # Handle case where years might be very small or zero to avoid errors
    if years > 1e-6: # Avoid division by zero or large exponents for near-zero years
        annual_return_pct = ((fv_total / price) ** (1 / years) - 1) * 100
    else:
        annual_return_pct = 0.0 # Or handle as appropriate (e.g., return total_return_pct)

    # Return the number of coupons as well
    return fv_total, total_return_pct, annual_return_pct, years, num_future_coupons


# --- Start of New/Modified YTM Calculation Logic ---

def get_coupon_dates(maturity_date: datetime.date, settlement_date: datetime.date) -> List[datetime.date]:
    """Generates all coupon dates from maturity back to before settlement."""
    dates = []
    current_date = maturity_date
    # Ensure we capture the maturity date itself if it's a coupon date
    if current_date >= settlement_date:
         dates.append(current_date)

    while True:
        prev_date = current_date - relativedelta(months=6)
        if prev_date < settlement_date:
            # Add the last coupon date *before* settlement if needed for accrued interest calc later
            # but don't add it to the list of *future* coupon dates for return calc.
            # The get_coupon_dates is also used by YTM calc, so need to handle both cases.
            # Let's adjust the logic slightly: generate all dates back to *before* settlement
            break # Stop before adding dates prior to settlement
        dates.append(prev_date)
        current_date = prev_date

    # The YTM function needs the coupon date *before* settlement too.
    # Let's refine this: get *all* dates, then filter later where needed.
    dates = []
    current_date = maturity_date
    while True:
        dates.append(current_date)
        current_date -= relativedelta(months=6)
        # Stop when the generated date is clearly before the period relevant to settlement
        if current_date < settlement_date - relativedelta(months=7):
             break

    return sorted(dates)

def calculate_actual_actual_days(start_date: datetime.date, end_date: datetime.date) -> int:
    """Calculates days using Actual/Actual convention."""
    return (end_date - start_date).days

def calculate_ytm_bey(clean_price: float, coupon_rate: float, maturity_date: datetime.date, face_value: float = 100) -> float:
    """
    Calculates Yield to Maturity using standard Bond Equivalent Yield (BEY) conventions.
    - T+1 Settlement
    - Actual/Actual Day Count
    - Semi-annual Compounding
    """
    today = datetime.date.today()
    # Assuming weekends/holidays are handled by simply adding days.
    # A more robust solution would use a business day calendar.
    settlement_date = today + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)

    coupon_amt = (coupon_rate / 100.0 * face_value) / 2.0

    # 1. Determine Coupon Dates
    all_coupon_dates = get_coupon_dates(maturity_date, settlement_date)
    future_coupon_dates = [d for d in all_coupon_dates if d > settlement_date]
    last_coupon_date = max(d for d in all_coupon_dates if d <= settlement_date)
    next_coupon_date = future_coupon_dates[0]

    # 2. Calculate Accrued Interest
    days_in_period = calculate_actual_actual_days(last_coupon_date, next_coupon_date)
    days_settle_to_next = calculate_actual_actual_days(settlement_date, next_coupon_date)
    days_accrued = days_in_period - days_settle_to_next

    if days_in_period == 0: # Avoid division by zero if settlement is on coupon date
        accrued_interest = 0.0
    else:
        accrued_interest = coupon_amt * (days_accrued / days_in_period)

    # 3. Calculate Dirty Price
    dirty_price = clean_price + accrued_interest

    # 4. Define the Pricing Function for the Solver
    cash_flows = [(d, coupon_amt) for d in future_coupon_dates]
    cash_flows[-1] = (cash_flows[-1][0], cash_flows[-1][1] + face_value) # Add principal

    if days_in_period == 0:
        w = 0 # Settlement is on coupon date
    else:
        w = days_settle_to_next / days_in_period

    def bond_price_func(yield_rate):
        pv = 0.0
        for i, (cf_date, cf_amount) in enumerate(cash_flows):
            # Discount factor based on semi-annual compounding and fractional period
            discount_factor = (1 + yield_rate / 2) ** (i + w)
            pv += cf_amount / discount_factor
        return pv - dirty_price

    # 5. Solve for Yield
    # Provide a reasonable starting guess (e.g., coupon rate / price)
    initial_guess = (coupon_rate / 100.0) / (clean_price / face_value)
    try:
        # Solve for the semi-annual yield (y/2), then multiply by 2 for BEY
        # Correction: Define function in terms of y (annual BEY), solver finds y directly.
        ytm_result = newton(bond_price_func, initial_guess, tol=1e-8, maxiter=100)
    except RuntimeError:
        # Fallback or error handling if solver fails
        # Try a simpler approximation or raise an error
        # For now, let's try the coupon rate as a fallback guess
        try:
            ytm_result = newton(bond_price_func, coupon_rate / 100.0, tol=1e-8, maxiter=100)
        except RuntimeError as e:
             raise RuntimeError(f"YTM calculation failed to converge: {e}")

    return ytm_result * 100 # Return as percentage

# --- End of New YTM Calculation Logic ---


# Keep the old calculate_ytm function name but have it call the new BEY logic
def calculate_ytm(price: float, coupon: float, maturity: datetime.date, face_value: float = 100) -> float:
    # This function now acts as a wrapper for the BEY calculation
    return calculate_ytm_bey(price, coupon, maturity, face_value)


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
        # Capture stdout/stderr to suppress potential debug prints from fetch_auction_yield
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            result = fetch_auction_yield(duration_code, datetime.datetime.today())

    if result and isinstance(result, tuple) and len(result) == 5:
        _, _, _, _, yield_value = result
        # Ensure yield_value is treated as a percentage if it comes from the API that way
        # Assuming the API returns yield as a percentage (e.g., 4.5 for 4.5%)
        return duration_code, yield_value / 100.0 # Convert percentage to decimal
    elif result and isinstance(result, (int, float)): # Handle case where only yield is returned
         return duration_code, result / 100.0
    else:
        # Attempt to provide more info if possible
        error_details = f.getvalue().strip()
        raise RuntimeError(f"Failed to fetch or parse benchmark yield for {duration_code}. API Output: '{error_details or 'None'}'")


def assign_grade(bond_ytm: float, benchmark: float) -> str:
    # Grading based on YTM difference now
    delta_bps = (bond_ytm - benchmark) * 100 # Difference in basis points
    if delta_bps >= 50: return "A+"
    elif delta_bps >= 25: return "A"
    elif delta_bps >= 10: return "A−"
    elif delta_bps >= 0: return "B+"
    elif delta_bps >= -9: return "B"
    elif delta_bps >= -24: return "B−"
    elif delta_bps >= -49: return "C"
    else: return "D"


def main():
    args = sys.argv[1:]

    if len(args) == 1 and args[0] == "-h":
        print_help(); sys.exit(0)
    elif len(args) == 1 and args[0] == "-v":
        print_version(); sys.exit(0)
    elif len(args) not in (3, 4):
        print("Error: Incorrect number of arguments.")
        print_help(); sys.exit(1)

    debug = "--debug" in args
    if debug:
        args.remove("--debug")

    try:
        coupon = float(args[0])
        maturity = datetime.datetime.strptime(args[1], "%m/%d/%Y").date()
        price = float(args[2])

        # Calculate YTM using the new BEY method
        ytm = calculate_ytm(price, coupon, maturity) # Calls calculate_ytm_bey

        # Calculate simplified returns (no reinvestment)
        # Capture the number of coupons returned
        fv, total_return, annual_return, years, num_coupons = calculate_returns(coupon, maturity, price) # Pass coupon rate

        duration_code, benchmark_yield = get_comparable_yield(maturity, debug=debug)
        benchmark_yield_pct = benchmark_yield * 100
        # Grade based on YTM vs Benchmark
        grade = assign_grade(ytm, benchmark_yield_pct)

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
            return f"{GREEN}{g}{RESET}" if g in ("A+", "A", "A−", "B+") else f"{ORANGE if g in ('B', 'B-') else RED}{g}{RESET}"

        def color_header(val: str) -> str:
            return f"\033[1m{CYAN}{val}{RESET}"


        print(color_header("\nBond Details"))
        print(f"Coupon: {coupon:.3f}% | Maturity: {maturity} | Price: ${price:.3f}")
        print(f"Settlement Date (T+1): {datetime.date.today() + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)}")
        print(color_header("\nCalculated Yield"))
        print(f"Yield to Maturity (BEY): {color_value(f'{ytm:.3f}%')}")

        print(color_header("\nBenchmark Comparison"))
        print(f"Comparable Treasury ({duration_code}): {color_value(f'{benchmark_yield_pct:.3f}%')}")
        ytm_diff_bps = (ytm - benchmark_yield_pct) * 100
        print(f"Spread vs Benchmark: {color_value(f'{ytm_diff_bps:+.1f} bps')}")
        print(f"Grade: {color_grade(grade)}") # Keep grade based on YTM

        # Updated: Display the simplified return calculation results (no reinvestment)
        # Include num_coupons in the header
        print(color_header(f"\nSimplified Return Projection ({num_coupons} coupons held, 0% reinvestment)")) # Updated header
        print(f"Holding Period: {years:.2f} years")
        print(f"Future Value (Principal + Coupons): {color_value(f'${fv:.2f}')}") # Updated label
        print(f"Total Return (Projected): {color_value(f'{total_return:.2f}%')}")
        print(f"Annualized Return (Projected): {color_value(f'{annual_return:.2f}%')}")

    except ValueError as e:
        print(f"Error: Invalid input format - {e}")
        print_help()
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if debug:
            import traceback
            traceback.print_exc() # Print full traceback in debug mode
        sys.exit(1)

if __name__ == "__main__":
    main()