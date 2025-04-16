#!/usr/bin/env python3
import sys
import datetime
import contextlib
import io
from typing import Tuple, List, Optional
from scipy.optimize import newton
from dateutil.relativedelta import relativedelta
from get_bond_yield import fetch_auction_yield  # Uses your existing API module
import math # Needed for convexity calculation if using direct formula

VERSION = "1.17" # Updated version
SETTLEMENT_LAG_DAYS = 1 # Use T+1 for Treasuries


def print_help():
    print(f"""\
bond_return_calc.py — this script evaluates a secondary market bond's return and evaluates a comparable recent public auction

USAGE:
    bond_return_calc.py <coupon> <maturity_MM/DD/YYYY> <price> [--debug]

DESCRIPTION:
    Computes Yield to Maturity (YTM) using standard Bond Equivalent Yield (BEY) conventions.
    BEY assumes semi-annual compounding (doubling the semi-annual yield) and uses an Actual/Actual day count for accrued interest. Settlement is T+1.
    Calculates Modified Duration and Convexity to assess interest rate risk.
    Compares the bond's YTM against an *interpolated* Treasury benchmark yield derived from the two closest standard Treasury maturities. This provides a more precise comparison point for the bond's specific maturity.
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
    Calculates simplified total return and a simple average annual return,
    assuming coupons are held (0% reinvestment).
    Note: Annual return is simple average, not compounded (CAGR).
    Returns: Tuple[float, float, float, float, int] -> fv_total, total_return_pct, simple_annual_return_pct, years, num_future_coupons
    """
    today = datetime.date.today()
    settlement_date = today + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)
    # Use settlement date for more accurate year calculation for return period
    years = (maturity - settlement_date).days / 365.25 # Years from settlement to maturity

    # Calculate total value of coupons received from settlement until maturity
    coupon_amt_semi_annual = (coupon_rate / 100.0 * face_value) / 2.0
    all_coupon_dates = get_coupon_dates(maturity, settlement_date) # Use existing helper
    future_coupon_dates = [d for d in all_coupon_dates if d > settlement_date]
    num_future_coupons = len(future_coupon_dates) # Number of coupons counted
    total_coupons_value = num_future_coupons * coupon_amt_semi_annual

    fv_total = face_value + total_coupons_value  # Final value = Principal + Sum of coupons received

    # Calculate total gain relative to the purchase price
    total_gain = fv_total - price
    if price <= 1e-9: # Avoid division by zero if price is zero or negative (already validated, but safe)
        total_return_pct = 0.0
        simple_annual_return_pct = 0.0
    else:
        total_return_pct = (total_gain / price) * 100

        # Calculate Simple Average Annual Return
        if years > 1e-6: # Avoid division by zero for very short periods
            simple_annual_return_pct = total_return_pct / years
        else:
            # If holding period is effectively zero, annual return is ambiguous.
            # Could return total_return_pct or 0. Let's return 0 for clarity.
            simple_annual_return_pct = 0.0

    # Return the number of coupons as well
    return fv_total, total_return_pct, simple_annual_return_pct, years, num_future_coupons

# --- End calculate_returns ---

# --- Start of YTM Calculation Logic ---

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

def calculate_ytm_bey(clean_price: float, coupon_rate: float, maturity_date: datetime.date, face_value: float = 100) -> Tuple[float, float, List[Tuple[datetime.date, float]], float]:
    """
    Calculates Yield to Maturity using standard Bond Equivalent Yield (BEY) conventions.
    Returns: Tuple[ytm_percentage, dirty_price, cash_flows, w_factor]
    """
    today = datetime.date.today()
    settlement_date = today + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)
    coupon_amt = (coupon_rate / 100.0 * face_value) / 2.0

    # 1. Determine Coupon Dates
    all_coupon_dates = get_coupon_dates(maturity_date, settlement_date)
    future_coupon_dates = [d for d in all_coupon_dates if d > settlement_date]
    if not future_coupon_dates:
        # Handle case where bond matures before or on settlement
        # Check if maturity is on settlement date
        if maturity_date == settlement_date:
             # Treat as zero-coupon for remaining calculations if needed, or raise specific error
             # For YTM, if price matches face value, yield is ambiguous/zero. If not, it's complex.
             # Let's raise an error for simplicity as YTM is ill-defined here.
             raise ValueError("Bond settles on maturity date. YTM calculation is not applicable.")
        else:
             raise ValueError("Bond has already matured or settles after maturity.")

    last_coupon_date = max(d for d in all_coupon_dates if d <= settlement_date)
    next_coupon_date = future_coupon_dates[0]

    # 2. Calculate Accrued Interest
    days_in_period = calculate_actual_actual_days(last_coupon_date, next_coupon_date)
    days_settle_to_next = calculate_actual_actual_days(settlement_date, next_coupon_date)
    # Ensure days_accrued is not negative if settlement is exactly on last_coupon_date
    days_accrued = max(0, days_in_period - days_settle_to_next)

    if days_in_period == 0: # Avoid division by zero if settlement is on coupon date
        accrued_interest = 0.0
        w = 0 # Settlement is on coupon date
    else:
        accrued_interest = coupon_amt * (days_accrued / days_in_period)
        w = days_settle_to_next / days_in_period # Fractional period factor

    # 3. Calculate Dirty Price
    dirty_price = clean_price + accrued_interest

    # 4. Define the Pricing Function for the Solver
    cash_flows = [(d, coupon_amt) for d in future_coupon_dates]
    cash_flows[-1] = (cash_flows[-1][0], cash_flows[-1][1] + face_value) # Add principal

    def bond_price_func(yield_rate):
        pv = 0.0
        for i, (cf_date, cf_amount) in enumerate(cash_flows):
            discount_factor = (1 + yield_rate / 2) ** (i + w)
            pv += cf_amount / discount_factor
        return pv - dirty_price

    # 5. Solve for Yield
    initial_guess = (coupon_rate / 100.0) / (clean_price / face_value)
    try:
        ytm_result_decimal = newton(bond_price_func, initial_guess, tol=1e-8, maxiter=100)
    except RuntimeError:
        try:
            ytm_result_decimal = newton(bond_price_func, coupon_rate / 100.0, tol=1e-8, maxiter=100)
        except RuntimeError as e:
             raise RuntimeError(f"YTM calculation failed to converge: {e}")

    # Return YTM, dirty price, cash flows, and w for duration/convexity
    return ytm_result_decimal * 100, dirty_price, cash_flows, w

# --- End of YTM Calculation Logic ---

# --- Start of Duration/Convexity Calculation ---
def calculate_duration_convexity(ytm_decimal: float, dirty_price: float, cash_flows: List[Tuple[datetime.date, float]], w_factor: float) -> Tuple[float, float]:
    """
    Calculates Modified Duration and Convexity for a bond.
    Assumes semi-annual compounding (k=2).
    """
    k = 2.0 # Compounding frequency (semi-annual)
    y = ytm_decimal # YTM as a decimal

    macaulay_numerator = 0.0
    convexity_numerator = 0.0

    # Iterate through cash flows to calculate weighted times and convexity terms
    for i, (cf_date, cf_amount) in enumerate(cash_flows):
        # Time factor 't' in periods (consistent with YTM calc)
        time_factor_periods = i + w_factor
        # Present value of this cash flow
        pv_cf = cf_amount / ((1 + y / k) ** time_factor_periods)

        # Macaulay Duration component
        macaulay_numerator += pv_cf * time_factor_periods

        # Convexity component
        # Note: Time factor 't' in the standard convexity formula is often in years.
        # Here, time_factor_periods is in *periods*. We need to adjust.
        # Let T = time_factor_periods / k (time in years)
        # Convexity formula term: CFt * T * (T + 1/k) / (1 + y/k)^(k*T + 2)
        # Simplified using periods: CFt * (i+w) * (i+w+1) / (1 + y/k)^(i+w+2)
        convexity_numerator += pv_cf * time_factor_periods * (time_factor_periods + 1) / ((1 + y / k)**2)


    # Calculate Macaulay Duration (in periods)
    macaulay_duration_periods = macaulay_numerator / dirty_price

    # Convert Macaulay Duration to years
    macaulay_duration_years = macaulay_duration_periods / k

    # Calculate Modified Duration (in years)
    modified_duration = macaulay_duration_years / (1 + y / k)

    # Calculate Convexity (in years^2)
    # Convexity = Sum[ pv_cf * t * (t+1) ] / ( DirtyPrice * (1+y/k)^2 ) -- when t is in periods
    convexity = convexity_numerator / (dirty_price) # Already divided by (1+y/k)^2 in loop

    # Sometimes convexity is divided by k^2 to scale it differently, but the formula above is common.
    # Let's stick to the formula derived from period-based calculation.

    return modified_duration, convexity

# --- End of Duration/Convexity Calculation ---


def _fetch_single_yield(duration_code: str, debug: bool) -> Optional[float]:
    """Helper to fetch and parse yield for a single duration code, suppressing internal API output unless debug=True."""
    result = None
    api_output_capture = io.StringIO() # To capture output when not in debug mode

    if debug:
        print(f"Fetching benchmark yield for {duration_code}...")
        # Allow fetch_auction_yield to print directly in debug mode
        try:
            result = fetch_auction_yield(duration_code, datetime.datetime.today())
        except Exception as e:
            # Still good to catch potential errors during fetch
             print(f"Error during fetch_auction_yield for {duration_code}: {e}", file=sys.stderr)
             # No return here, let the result processing handle None
    else:
        # Use contextlib to capture stdout and stderr when not in debug mode
        with contextlib.redirect_stdout(api_output_capture), contextlib.redirect_stderr(api_output_capture):
            try:
                result = fetch_auction_yield(duration_code, datetime.datetime.today())
            except Exception as e:
                 # In non-debug, we might just want to know it failed without printing the exception
                 # Or optionally log it differently. For now, let it pass to result check.
                 pass # Error will be implicitly handled by result being None

    # Process the result
    if result and isinstance(result, tuple) and len(result) == 5:
        _, _, _, _, yield_value = result
        # This print is already controlled by the outer debug check
        if debug: print(f"Fetched {duration_code} yield: {yield_value}%")
        return yield_value / 100.0 # Convert percentage to decimal
    elif result and isinstance(result, (int, float)):
         # This print is already controlled by the outer debug check
        if debug: print(f"Fetched {duration_code} yield: {result}%")
        return result / 100.0
    else:
        # Only print the failure details if in debug mode
        if debug:
            error_details = api_output_capture.getvalue().strip() # Get captured output if any
            print(f"Failed to fetch or parse benchmark yield for {duration_code}. API Output: '{error_details or 'None'}'")
        return None


def get_comparable_yield(maturity_date: datetime.date, debug: bool = False) -> Tuple[str, float, Optional[float], Optional[float]]:
    """
    Calculates a comparable benchmark yield by interpolating between
    the two closest standard Treasury maturities. Returns the yields used.
    """
    today = datetime.date.today()
    years_remaining = (maturity_date - today).days / 365.25

    lower_term = None
    upper_term = None
    lower_yield_val: Optional[float] = None
    upper_yield_val: Optional[float] = None

    # Find surrounding terms
    SORTED_TERM_YEARS = [
        ("2Y", 2),
        ("3Y", 3),
        ("5Y", 5),
        ("7Y", 7),
        ("10Y", 10),
        ("20Y", 20),
        ("30Y", 30)
    ]
    for i, (term_code, term_years) in enumerate(SORTED_TERM_YEARS):
        if years_remaining <= term_years:
            upper_term = (term_code, term_years)
            if i > 0:
                lower_term = SORTED_TERM_YEARS[i-1]
            break
    else: # If years_remaining is greater than the longest term
        lower_term = SORTED_TERM_YEARS[-1]
        upper_term = None # Use longest term directly


    # Handle edge cases or direct matches
    if lower_term is None: # Maturity is shorter than the shortest benchmark term
        term_code, _ = SORTED_TERM_YEARS[0]
        yield_val = _fetch_single_yield(term_code, debug)
        if yield_val is None:
            raise RuntimeError(f"Failed to fetch required benchmark yield for {term_code}")
        return f"{term_code} (Direct)", yield_val, None, None # No interpolation bounds

    if upper_term is None: # Maturity is longer than the longest benchmark term
        term_code, _ = lower_term
        yield_val = _fetch_single_yield(term_code, debug)
        if yield_val is None:
            raise RuntimeError(f"Failed to fetch required benchmark yield for {term_code}")
        return f"{term_code} (Direct)", yield_val, None, None # No interpolation bounds

    lower_code, lower_years = lower_term
    upper_code, upper_years = upper_term

    # If maturity exactly matches a benchmark term
    if abs(years_remaining - lower_years) < 1e-6:
        yield_val = _fetch_single_yield(lower_code, debug)
        if yield_val is None: raise RuntimeError(f"Failed to fetch required benchmark yield for {lower_code}")
        return f"{lower_code} (Direct)", yield_val, None, None # No interpolation bounds
    if abs(years_remaining - upper_years) < 1e-6:
        yield_val = _fetch_single_yield(upper_code, debug)
        if yield_val is None: raise RuntimeError(f"Failed to fetch required benchmark yield for {upper_code}")
        return f"{upper_code} (Direct)", yield_val, None, None # No interpolation bounds

    # Fetch yields for interpolation
    lower_yield_val = _fetch_single_yield(lower_code, debug)
    upper_yield_val = _fetch_single_yield(upper_code, debug)

    if lower_yield_val is None or upper_yield_val is None:
        # Fallback: If interpolation fails, try to get the single closest benchmark yield
        closest_code = upper_code if (years_remaining - lower_years) > (upper_years - years_remaining) else lower_code
        fallback_yield = _fetch_single_yield(closest_code, debug)
        if fallback_yield is None:
            raise RuntimeError(f"Failed to fetch required benchmark yields for interpolation ({lower_code}/{upper_code}) and fallback ({closest_code})")
        print(f"Warning: Interpolation failed (missing data?). Using closest benchmark {closest_code} yield.", file=sys.stderr)
        # Return fallback yield, indicate bounds used if available
        return f"{closest_code} (Fallback)", fallback_yield, lower_yield_val, upper_yield_val


    # Perform linear interpolation
    interp_fraction = (years_remaining - lower_years) / (upper_years - lower_years)
    interpolated_yield = lower_yield_val + interp_fraction * (upper_yield_val - lower_yield_val)

    label = f"Interpolated ({lower_code}/{upper_code})"
    # Return interpolated yield and the bounds used
    return label, interpolated_yield, lower_yield_val, upper_yield_val


# Updated assign_grade to consider maturity and use F as lowest grade
def assign_grade(bond_ytm: float, benchmark: float, years_to_maturity: float) -> str:
    """Assigns a grade based on spread vs benchmark, adjusted for maturity."""
    delta_bps = (bond_ytm - benchmark) * 100 # Difference in basis points

    # Define thresholds based on maturity ranges (example ranges, adjust as needed)
    if years_to_maturity <= 3: # Short-term
        if delta_bps >= 30: return "A+"
        elif delta_bps >= 15: return "A"
        elif delta_bps >= 5: return "A−"
        elif delta_bps >= -2: return "B+"
        elif delta_bps >= -10: return "B"
        elif delta_bps >= -20: return "B−"
        elif delta_bps >= -35: return "C"
        else: return "F" # Changed from D
    elif years_to_maturity <= 10: # Medium-term
        if delta_bps >= 50: return "A+"
        elif delta_bps >= 25: return "A"
        elif delta_bps >= 10: return "A−"
        elif delta_bps >= 0: return "B+"
        elif delta_bps >= -9: return "B"
        elif delta_bps >= -24: return "B−"
        elif delta_bps >= -49: return "C"
        else: return "F" # Changed from D
    else: # Long-term (> 10 years)
        if delta_bps >= 70: return "A+"
        elif delta_bps >= 40: return "A"
        elif delta_bps >= 20: return "A−"
        elif delta_bps >= 5: return "B+"
        elif delta_bps >= -5: return "B"
        elif delta_bps >= -15: return "B−"
        elif delta_bps >= -30: return "C"
        else: return "F" # Changed from D


def main():
    args = sys.argv[1:]

    # --- Argument Parsing and Initial Validation ---
    if len(args) == 1 and args[0] == "-h":
        print_help(); sys.exit(0)
    elif len(args) == 1 and args[0] == "-v":
        print_version(); sys.exit(0)

    debug = "--debug" in args
    if debug:
        args = [arg for arg in args if arg != "--debug"] # More robust removal

    if len(args) != 3:
        print("Error: Incorrect number of arguments. Expected: <coupon> <maturity_MM/DD/YYYY> <price>")
        print_help(); sys.exit(1)

    coupon_str, maturity_str, price_str = args
    # --- End Argument Parsing ---

    try:
        # --- Input Validation ---
        try:
            coupon = float(coupon_str)
            if coupon < 0:
                raise ValueError("Coupon rate cannot be negative.")
        except ValueError:
            raise ValueError(f"Invalid coupon rate format: '{coupon_str}'. Please use a number.")

        try:
            maturity = datetime.datetime.strptime(maturity_str, "%m/%d/%Y").date()
            if maturity <= datetime.date.today():
                 # Allow maturity on today's date for potential zero-day calcs, but not past.
                 # YTM function will handle settlement vs maturity check.
                 pass # Or raise ValueError("Maturity date must be in the future.") if preferred
        except ValueError:
            raise ValueError(f"Invalid maturity date format: '{maturity_str}'. Please use MM/DD/YYYY.")

        try:
            price = float(price_str)
            if price <= 0:
                raise ValueError("Price must be a positive number.")
        except ValueError:
            raise ValueError(f"Invalid price format: '{price_str}'. Please use a number.")
        # --- End Input Validation ---


        # --- Core Calculations ---
        # Calculate YTM, get necessary components for Duration/Convexity
        ytm_pct, dirty_price_calc, cash_flows_calc, w_factor_calc = calculate_ytm_bey(price, coupon, maturity)
        ytm_decimal = ytm_pct / 100.0

        # Calculate Duration and Convexity
        mod_duration, convexity_val = calculate_duration_convexity(ytm_decimal, dirty_price_calc, cash_flows_calc, w_factor_calc)

        # Calculate simplified returns (no reinvestment) - now returns simple annual return
        fv, total_return, simple_annual_return, years, num_coupons = calculate_returns(coupon, maturity, price) # Renamed variable

        # Get interpolated benchmark yield and bounds
        benchmark_label, benchmark_yield, lower_bm_yield, upper_bm_yield = get_comparable_yield(maturity, debug=debug)
        benchmark_yield_pct = benchmark_yield * 100

        # Grade based on YTM vs Benchmark, now passing 'years'
        grade = assign_grade(ytm_pct, benchmark_yield_pct, years) # Pass years
        ytm_diff_bps = (ytm_pct - benchmark_yield_pct) * 100 # Calculate spread

        # Determine maturity category for explanation
        if years <= 3:
            maturity_context = "short-term maturity"
        elif years <= 10:
            maturity_context = "medium-term maturity"
        else:
            maturity_context = "long-term maturity"

        # Construct explanation string
        grade_explanation = f"(due to {ytm_diff_bps:+.1f} bps spread for {maturity_context})"
        # --- End Core Calculations ---


        # --- Output Formatting ---
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
            if g in ("A+", "A", "A−", "B+"):
                return f"{GREEN}{g}{RESET}"
            elif g in ('B', 'B-'):
                return f"{ORANGE}{g}{RESET}"
            else: # C, F, or any other lower grade
                return f"{RED}{g}{RESET}"

        def color_header(val: str) -> str:
            return f"\033[1m{CYAN}{val}{RESET}"
        # --- End Output Formatting ---


        # --- Print Results ---
        print(color_header("\nBond Details"))
        print(f"Coupon: {coupon:.3f}% | Maturity: {maturity} | Price: ${price:.3f}")
        print(f"Settlement Date (T+1): {datetime.date.today() + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)}")
        print(f"Dirty Price: ${dirty_price_calc:.5f}")

        print(color_header("\nCalculated Yield & Risk"))
        print(f"Yield to Maturity (BEY): {color_value(f'{ytm_pct:.3f}%')}")
        print(f"Modified Duration: {color_value(f'{mod_duration:.4f}')}")
        print(f"Convexity: {color_value(f'{convexity_val:.4f}')}")

        print(color_header("\nBenchmark Comparison"))
        # ... (benchmark printing logic remains the same) ...
        if lower_bm_yield is not None and upper_bm_yield is not None and "Interpolated" in benchmark_label:
             codes_part = benchmark_label.split("(")[1].split(")")[0]
             lower_code, upper_code = codes_part.split("/")
             print(f"  Lower Bound ({lower_code}): {color_value(f'{lower_bm_yield*100:.3f}%')}")
             print(f"  Upper Bound ({upper_code}): {color_value(f'{upper_bm_yield*100:.3f}%')}")
             print(f"Interpolated Benchmark ({years:.2f} yrs): {color_value(f'{benchmark_yield_pct:.3f}%')}")
        elif "Fallback" in benchmark_label:
             codes_part = benchmark_label.split("(")[1].split(")")[0]
             print(f"Used Fallback Benchmark ({codes_part}): {color_value(f'{benchmark_yield_pct:.3f}%')}")
             if lower_bm_yield is not None: print(f"  (Lower bound fetch attempt: {lower_bm_yield*100:.3f}%)")
             if upper_bm_yield is not None: print(f"  (Upper bound fetch attempt: {upper_bm_yield*100:.3f}%)")
        else: # Direct match or edge case
             print(f"Comparable Treasury ({benchmark_label}): {color_value(f'{benchmark_yield_pct:.3f}%')}")

        print(f"Spread vs Benchmark: {color_value(f'{ytm_diff_bps:+.1f} bps')}")
        print(f"Grade: {color_grade(grade)} {grade_explanation}")

        print(color_header(f"\nSimplified Return Projection ({num_coupons} coupons held, 0% reinvestment)"))
        # Use settlement date for holding period start
        settlement_dt_print = datetime.date.today() + datetime.timedelta(days=SETTLEMENT_LAG_DAYS)
        print(f"Holding Period: {years:.2f} years (from {settlement_dt_print} to {maturity})") # Clarified period
        print(f"Future Value (Principal + Coupons): {color_value(f'${fv:.2f}')}")
        print(f"Total Return (Projected): {color_value(f'{total_return:.2f}%')}")
        # Updated label for clarity
        print(f"Simple Avg Annual Return (Projected): {color_value(f'{simple_annual_return:.2f}%')}") # Updated label and variable

        # --- End Print Results ---

    except ValueError as e: # Catch specific input validation errors
        print(f"Input Error: {e}")
        print_help()
        sys.exit(1)
    except RuntimeError as e: # Catch convergence/fetch errors
        print(f"Calculation Error: {e}")
        sys.exit(1)
    except Exception as e: # Catch unexpected errors
        print(f"An unexpected error occurred: {e}")
        if debug:
            import traceback
            traceback.print_exc() # Print full traceback in debug mode
        sys.exit(1)

if __name__ == "__main__":
    main()