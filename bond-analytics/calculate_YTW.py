#!/usr/bin/env python3

import datetime
import argparse
import sys
import json
from typing import Union, List, Tuple

from dateutil.relativedelta import relativedelta
VERSION = "1.0.3"


def _parse_date_input(date_str: str) -> datetime.date:
    """
    Parses a date string, assuming 20xx for 2-digit years.
    Handles both MM/DD/YYYY and MM/DD/YY.
    """
    if not isinstance(date_str, str):
        raise TypeError("date_str must be a string.")

    date_str = date_str.strip()
    parts = date_str.split('/')
    
    if len(parts) == 3 and len(parts[2]) == 2:
        # It's a 2-digit year. Prepend '20'.
        try:
            # Ensure the 2-digit part is a number
            int(parts[2])
            corrected_date_str = f"{parts[0]}/{parts[1]}/20{parts[2]}"
            return datetime.datetime.strptime(corrected_date_str, '%m/%d/%Y').date()
        except (ValueError, TypeError):
            # Re-raise with original string for a better error message
            raise ValueError(f"Date '{date_str}' is not a valid date in MM/DD/YY format.")
    else:
        # Assume it's a 4-digit year or invalid format
        try:
            return datetime.datetime.strptime(date_str, '%m/%d/%Y').date()
        except ValueError:
            raise ValueError(f"Date '{date_str}' is not in a valid MM/DD/YYYY or MM/DD/YY format.")


def calculate_ytw_bey(
    coupon_rate: float,
    maturity_date: Union[str, datetime.date],
    price: float,
    settlement_date: Union[str, datetime.date, None] = None,
    face_value: float = 100.0
) -> float:
    """
    Calculates Yield to Maturity using standard Bond Equivalent Yield (BEY) conventions.
    This is an industry-standard model that accounts for accrued interest.
    """
    # 1. Setup and Date Parsing
    if isinstance(maturity_date, str):
        maturity_date = _parse_date_input(maturity_date)

    if settlement_date is None:
        settlement_date = datetime.date.today() + datetime.timedelta(days=1)
    elif isinstance(settlement_date, str):
        settlement_date = _parse_date_input(settlement_date)

    if maturity_date <= settlement_date:
        raise ValueError("Maturity date must be after settlement date")

    coupon_amt = (coupon_rate / 100.0 * face_value) / 2.0

    # 2. Determine Coupon Dates
    all_coupon_dates = get_coupon_dates(maturity_date, settlement_date)
    future_coupon_dates = [d for d in all_coupon_dates if d > settlement_date]

    if not future_coupon_dates:
        raise ValueError("Bond settles on or after the final coupon date.")

    last_coupon_date_candidates = [d for d in all_coupon_dates if d <= settlement_date]
    if not last_coupon_date_candidates:
        # This can happen for a new bond. We approximate the previous coupon date.
        last_coupon_date = future_coupon_dates[0] - relativedelta(months=6)
    else:
        last_coupon_date = max(last_coupon_date_candidates)

    next_coupon_date = future_coupon_dates[0]

    # 3. Calculate Accrued Interest and 'w' factor
    days_in_period = calculate_actual_actual_days(last_coupon_date, next_coupon_date)
    days_settle_to_next = calculate_actual_actual_days(settlement_date, next_coupon_date)
    days_accrued = days_in_period - days_settle_to_next

    if days_in_period == 0:
        accrued_interest = 0.0
        w = 0.0
    else:
        accrued_interest = coupon_amt * (days_accrued / days_in_period)
        w = days_settle_to_next / days_in_period

    # 4. Calculate Dirty Price (Full price)
    dirty_price = price + accrued_interest

    # 5. Define Cash Flows
    cash_flows: List[Tuple[datetime.date, float]] = [(d, coupon_amt) for d in future_coupon_dates]
    cash_flows[-1] = (cash_flows[-1][0], cash_flows[-1][1] + face_value)

    # 6. Newton-Raphson Solver
    yield_guess = coupon_rate / 100.0
    tolerance = 1e-8
    max_iterations = 200

    for _ in range(max_iterations):
        semi_annual_yield = yield_guess / 2.0
        
        present_value = 0.0
        derivative = 0.0
        
        for i, (cf_date, cf_amount) in enumerate(cash_flows):
            time_factor = i + w
            discount_factor = (1 + semi_annual_yield) ** time_factor
            
            present_value += cf_amount / discount_factor
            
            # Derivative component for this cash flow
            derivative += -cf_amount * time_factor * ((1 + semi_annual_yield) ** (-time_factor - 1))

        # Adjust derivative for chain rule (d/dy where y is annual yield)
        derivative /= 2.0
        
        f = present_value - dirty_price
        
        if abs(f) < tolerance:
            return yield_guess * 100.0
        
        if derivative == 0:
             raise RuntimeError("YTM calculation failed: derivative is zero.")

        yield_guess = yield_guess - f / derivative
        
        if yield_guess < -1.0: # Allow for negative yields but cap it
            yield_guess = -1.0

    raise RuntimeError("YTM calculation failed to converge.")


def get_coupon_dates(maturity_date: datetime.date, settlement_date: datetime.date) -> List[datetime.date]:
    """Generates all coupon dates from maturity back to before settlement."""
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


def get_interactive_inputs():
    """Prompt user for bond details interactively."""
    print("Entering interactive mode to calculate bond YTW.")
    print("Press Ctrl+C to exit at any time.")

    # Prompt for coupon
    while True:
        try:
            coupon_str = input("Enter annual coupon rate (e.g., 4.75): ")
            coupon = float(coupon_str)
            break
        except ValueError:
            print("Invalid input. Please enter a number for the coupon rate.")

    # Prompt for maturity
    while True:
        try:
            maturity_str = input("Enter maturity date (MM/DD/YYYY or MM/DD/YY): ")
            _parse_date_input(maturity_str)  # Validate input
            maturity = maturity_str
            break
        except ValueError as e:
            print(f"Error: {e}")

    # Prompt for price
    while True:
        try:
            price_str = input("Enter clean price (e.g., 96.875): ")
            price = float(price_str)
            break
        except ValueError:
            print("Invalid input. Please enter a number for the price.")

    # Prompt for settlement (optional)
    while True:
        try:
            settlement_str = input("Enter settlement date (MM/DD/YYYY or MM/DD/YY, or press Enter for T+1): ")
            if not settlement_str:
                settlement = None
                break
            _parse_date_input(settlement_str)  # Validate input
            settlement = settlement_str
            break
        except ValueError as e:
            print(f"Error: {e}")

    # Prompt for face value (optional)
    while True:
        try:
            facevalue_str = input("Enter face value (or press Enter for 100.0): ")
            if not facevalue_str:
                facevalue = 100.0
                break
            facevalue = float(facevalue_str)
            break
        except ValueError:
            print("Invalid input. Please enter a number for the face value.")

    return coupon, maturity, price, settlement, facevalue


def main():
    """Main function to parse arguments and calculate YTW."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    ORANGE = "\033[33m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description=f"""{GREEN}{BOLD}Calculate Yield to Maturity (YTM) for a bond.{RESET}

{CYAN}{BOLD}Overview:{RESET}
  This script calculates a bond's Yield to Maturity (YTM) using the industry-standard
  Bond Equivalent Yield (BEY) method. This method accurately accounts for accrued
  interest and semi-annual coupon payments.

{CYAN}{BOLD}How it works:{RESET}
  The script can be run in two modes:
  1. {BOLD}Command-line:{RESET} Provide coupon, maturity, and price as arguments.
  2. {BOLD}Interactive:{RESET} Run the script without the main arguments (or with flags like --showjson)
     to be prompted for the bond's details.

{CYAN}{BOLD}Arguments:{RESET}
  {BOLD}Positional Arguments (required for command-line mode):{RESET}
    coupon          Annual coupon rate as a percentage (e.g., 4.75)
    maturity        Maturity date in MM/DD/YYYY or MM/DD/YY format
    price           Clean price of the bond (e.g., 96.875)

  {BOLD}Optional Flags:{RESET}
    --settlement    Specify a settlement date (defaults to T+1).
    --facevalue     Specify the bond's face value (defaults to 100).
    --showjson      Output the results in JSON format.

{CYAN}{BOLD}Usage Examples:{RESET}
  - Calculate YTM for a bond using default T+1 settlement date:
    {YELLOW}python calculate_YTW.py 4.75 02/15/2045 96.875{RESET}

  - Calculate YTM with a specific settlement date:
    {YELLOW}python calculate_YTW.py 4.75 02/15/2045 96.875 --settlement 05/01/2024{RESET}

  - Get JSON output (will enter interactive mode if bond details are not provided):
    {YELLOW}python calculate_YTW.py --showjson{RESET}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Positional arguments are now optional (nargs='?') to allow checking for their presence.
    parser.add_argument("coupon", type=float, nargs='?', default=None, help="Annual coupon rate as a percentage (e.g., 4.75)")
    parser.add_argument("maturity", type=str, nargs='?', default=None, help="Maturity date in MM/DD/YYYY or MM/DD/YY format")
    parser.add_argument("price", type=float, nargs='?', default=None, help="Clean price of the bond (e.g., 96.875)")
    parser.add_argument("--settlement", type=str, help="Settlement date in MM/DD/YYYY format (default: T+1)")
    parser.add_argument("--showjson", action="store_true", help="Output results in JSON format.")
    parser.add_argument("--facevalue", type=float, default=100.0, help="Face value of the bond (default: 100.0)")
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {VERSION}')

    args = parser.parse_args()

    # --- Get bond parameters, either from args or interactive prompt ---
    if args.coupon is None or args.maturity is None or args.price is None:
        try:
            coupon, maturity, price, settlement, facevalue = get_interactive_inputs()
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
    else:
        coupon = args.coupon
        maturity = args.maturity
        price = args.price
        settlement = args.settlement
        facevalue = args.facevalue

    # --- Perform calculations and output results ---
    try:
        # Validate date formats before passing to function
        _parse_date_input(maturity)
        if settlement:
            _parse_date_input(settlement)

        ytw_bey = calculate_ytw_bey(
            coupon_rate=coupon,
            maturity_date=maturity,
            price=price,
            settlement_date=settlement,
            face_value=facevalue
        )

        # Output based on --showjson flag
        if args.showjson:
            if settlement:
                settlement_date_obj = _parse_date_input(settlement)
            else:
                settlement_date_obj = datetime.date.today() + datetime.timedelta(days=1)
            maturity_date_obj = _parse_date_input(maturity)
            json_output = {
                "Coupon": format(coupon, 'g'),
                "Maturity": maturity_date_obj.strftime('%m/%d/%Y'),
                "Price": format(price, 'g'),
                "YTW": f"{ytw_bey:.3f}",
                "Date": settlement_date_obj.strftime('%Y-%m-%d')
            }
            # Generate JSON with indent=2, then prefix each line with 4 spaces
            # to match the desired output format for easy copy-pasting.
            json_str = json.dumps(json_output, indent=2)
            indented_json_str = "\n".join(["    " + line for line in json_str.splitlines()])
            print(indented_json_str)
        else:
            # print(f"YTW (ACT/ACT Method): {ytw:.3f}%")
            # print(f"{YELLOW}YTW (Discount Yield Method): {alt_ytw:.3f}%{RESET}")
            print(f"\n{GREEN}YTW (Industry BEY Method): {ytw_bey:.3f}%{RESET}")

    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
