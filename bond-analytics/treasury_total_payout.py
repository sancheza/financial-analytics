#!/usr/bin/env python3
# ==============================================================================
# Script Name   : treasury_total_payout.py
# Description   : Calculates the total nominal cumulative cash payout (principal
#                 returned plus all remaining coupon distributions) for a U.S.
#                 Treasury bond allocation. Supports command-line execution or
#                 interactive prompt with pick-lists.
# Author        : Fixed Income Analytics Utility
# ==============================================================================

"""Treasury Total Nominal Payout Calculator.

Computes the total cash return of a fixed-principal U.S. Treasury security
across its full term to maturity based on total capital outlay, clean price,
annual coupon rate, and remaining semi-annual coupon payments. Default settlement
adheres to standard T+1 business day conventions under SEC Rule 15c6-1.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from typing import Optional

__version__ = "1.1.0"

# Common preset options for interactive selection
COMMON_OUTLAYS = [10_000.0, 50_000.0, 100_000.0, 500_000.0, 1_000_000.0]
COMMON_COUPONS = [3.875, 4.000, 4.125, 4.250, 4.375, 4.500, 4.625, 4.750, 5.000]
COMMON_MATURITIES = [
    datetime.date(2034, 8, 15),
    datetime.date(2035, 2, 15),
    datetime.date(2035, 8, 15),
    datetime.date(2036, 2, 15),
    datetime.date(2036, 8, 15),
    datetime.date(2037, 5, 15),
]


def get_default_settlement() -> datetime.date:
    """Returns the next business day (T+1) for standard U.S. Treasury settlement.

    Accounts for weekend rollovers (Friday trades settle Monday, Saturday/Sunday
    settle Monday).
    """
    settle = datetime.date.today() + datetime.timedelta(days=1)
    if settle.weekday() == 5:  # Saturday -> Monday
        settle += datetime.timedelta(days=2)
    elif settle.weekday() == 6:  # Sunday -> Monday
        settle += datetime.timedelta(days=1)
    return settle


def count_remaining_coupons(settlement: datetime.date, maturity: datetime.date, frequency: int = 2) -> int:
    """Calculates remaining coupon payments working backward from maturity.

    Args:
        settlement: Settlement date of the purchase.
        maturity: Stated maturity date of the Treasury bond.
        frequency: Coupon payments per year (default: 2 for semi-annual).

    Returns:
        Total integer count of remaining coupon payments strictly after settlement.
    """
    if maturity <= settlement:
        return 0

    months_step = 12 // frequency
    count = 0
    curr_year = maturity.year
    curr_month = maturity.month
    day = maturity.day

    while True:
        try:
            curr_date = datetime.date(curr_year, curr_month, day)
        except ValueError:
            curr_date = datetime.date(curr_year, curr_month, 28)

        if curr_date <= settlement:
            break

        count += 1

        curr_month -= months_step
        if curr_month <= 0:
            curr_month += 12
            curr_year -= 1

    return count


def calculate_payout(
    outlay: float,
    clean_price: float,
    coupon_rate_pct: float,
    coupons_remaining: int,
    lot_size: int = 1000,
) -> dict[str, float]:
    """Calculates both fractional and discrete round-lot nominal cash returns.

    Args:
        outlay: Total capital available for purchase.
        clean_price: Quoted clean price per $100 face value.
        coupon_rate_pct: Annual coupon rate expressed as a percentage (e.g. 4.625).
        coupons_remaining: Total count of remaining coupon distributions.
        lot_size: Secondary market note increment (default: $1,000).

    Returns:
        Dictionary containing par values, total coupons, and final cash payouts.
    """
    coupon_rate_dec = coupon_rate_pct / 100.0

    # 1. Exact / Fractional allocation (100% capital deployed)
    par_fractional = outlay / (clean_price / 100.0)
    coupon_pmt_fractional = par_fractional * (coupon_rate_dec / 2.0)
    total_coupons_fractional = coupons_remaining * coupon_pmt_fractional
    total_payout_fractional = par_fractional + total_coupons_fractional

    # 2. Discrete lot allocation ($1,000 standard market increments)
    cost_per_note = (clean_price / 100.0) * lot_size
    notes_bought = int(outlay // cost_per_note)
    par_discrete = notes_bought * lot_size
    actual_spent = notes_bought * cost_per_note
    unspent_cash = outlay - actual_spent
    coupon_pmt_discrete = par_discrete * (coupon_rate_dec / 2.0)
    total_coupons_discrete = coupons_remaining * coupon_pmt_discrete
    total_payout_discrete = par_discrete + total_coupons_discrete + unspent_cash

    return {
        "par_fractional": par_fractional,
        "total_coupons_fractional": total_coupons_fractional,
        "total_payout_fractional": total_payout_fractional,
        "notes_bought": float(notes_bought),
        "par_discrete": float(par_discrete),
        "actual_spent": actual_spent,
        "unspent_cash": unspent_cash,
        "total_coupons_discrete": total_coupons_discrete,
        "total_payout_discrete": total_payout_discrete,
    }


def prompt_pick_or_custom(prompt_name: str, options: list) -> str:
    """Presents a numbered pick list of options alongside an option for custom input."""
    print(f"\nSelect {prompt_name}:")
    for idx, opt in enumerate(options, 1):
        if isinstance(opt, datetime.date):
            formatted = opt.strftime("%Y-%m-%d")
        elif isinstance(opt, float) and opt >= 1000:
            formatted = f"${opt:,.2f}"
        elif isinstance(opt, float):
            formatted = f"{opt:.3f}%"
        else:
            formatted = str(opt)
        print(f"  [{idx}] {formatted}")
    print(f"  [{len(options) + 1}] Enter custom value")

    while True:
        choice = input(f"Choice (1-{len(options) + 1}): ").strip()
        if choice.isdigit():
            c = int(choice)
            if 1 <= c <= len(options):
                selected = options[c - 1]
                if isinstance(selected, datetime.date):
                    return selected.strftime("%Y-%m-%d")
                return str(selected)
            if c == len(options) + 1:
                return input(f"Enter custom {prompt_name}: ").strip()
        print("Invalid selection. Try again.")


def run_interactive() -> tuple[float, float, float, int, datetime.date, datetime.date]:
    """Runs interactive terminal prompts with pick lists."""
    print("=" * 60)
    print(" U.S. TREASURY NOMINAL PAYOUT CALCULATOR (INTERACTIVE MODE)")
    print("=" * 60)

    # 1. Outlay
    outlay_str = prompt_pick_or_custom("Investment Capital Outlay", COMMON_OUTLAYS)
    outlay = float(outlay_str.replace("$", "").replace(",", ""))

    # 2. Coupon
    coupon_str = prompt_pick_or_custom("Annual Coupon Rate (%)", COMMON_COUPONS)
    coupon = float(coupon_str.replace("%", "").strip())

    # 3. Clean Price
    clean_price_str = input("\nEnter Clean Price per $100 par (e.g. 96.469): ").strip()
    clean_price = float(clean_price_str)

    # 4. Dates
    default_settle_str = get_default_settlement().strftime("%Y-%m-%d")
    settle_input = input(f"\nEnter Settlement Date [YYYY-MM-DD] (default T+1: {default_settle_str}): ").strip()
    settle_str = settle_input if settle_input else default_settle_str
    settlement = datetime.datetime.strptime(settle_str, "%Y-%m-%d").date()

    mat_str = prompt_pick_or_custom("Maturity Date", COMMON_MATURITIES)
    maturity = datetime.datetime.strptime(mat_str, "%Y-%m-%d").date()

    coupons_remaining = count_remaining_coupons(settlement, maturity)
    return outlay, clean_price, coupon, coupons_remaining, settlement, maturity


def main() -> None:
    """Main CLI parser and execution flow."""
    parser = argparse.ArgumentParser(
        prog="treasury_total_payout.py",
        description="Compute total nominal cash returned from a U.S. Treasury investment.",
        epilog="If no positional/optional bond arguments are provided, interactive mode launches.",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--outlay", type=float, help="Total investment dollar amount (e.g., 1000000)"
    )
    parser.add_argument(
        "--price", type=float, help="Clean price per $100 par (e.g., 96.469)"
    )
    parser.add_argument(
        "--coupon", type=float, help="Annual coupon percentage (e.g., 4.625)"
    )
    parser.add_argument(
        "--settlement",
        type=str,
        default=get_default_settlement().strftime("%Y-%m-%d"),
        help="Settlement date YYYY-MM-DD (defaults to next business day T+1)",
    )
    parser.add_argument(
        "--maturity", type=str, help="Maturity date YYYY-MM-DD (e.g., 2036-08-15)"
    )
    parser.add_argument(
        "--coupons-remaining",
        type=int,
        help="Directly specify remaining coupon count (overrides maturity calculation)",
    )

    args = parser.parse_args()

    # Determine execution mode: CLI flags or Interactive pick list
    if args.outlay is not None and args.price is not None and args.coupon is not None:
        outlay = args.outlay
        clean_price = args.price
        coupon = args.coupon
        settlement = datetime.datetime.strptime(args.settlement, "%Y-%m-%d").date()

        if args.coupons_remaining is not None:
            coupons_remaining = args.coupons_remaining
            maturity_desc = f"N/A ({coupons_remaining} coupons manually specified)"
        elif args.maturity is not None:
            maturity = datetime.datetime.strptime(args.maturity, "%Y-%m-%d").date()
            coupons_remaining = count_remaining_coupons(settlement, maturity)
            maturity_desc = maturity.strftime("%Y-%m-%d")
        else:
            sys.exit("Error: Must provide either --maturity or --coupons-remaining.")
    else:
        outlay, clean_price, coupon, coupons_remaining, settlement, maturity = run_interactive()
        maturity_desc = maturity.strftime("%Y-%m-%d")

    res = calculate_payout(
        outlay=outlay,
        clean_price=clean_price,
        coupon_rate_pct=coupon,
        coupons_remaining=coupons_remaining,
    )

    print("\n" + "=" * 60)
    print(" TREASURY CASH FLOW RESULTS")
    print("=" * 60)
    print(f"Settlement Date      : {settlement}")
    print(f"Maturity Date        : {maturity_desc}")
    print(f"Remaining Coupons    : {coupons_remaining} semi-annual payments")
    print(f"Clean Price          : {clean_price:.3f}")
    print(f"Annual Coupon Rate   : {coupon:.3f}%\n")

    print("-" * 60)
    print("DISCRETE ROUND-LOT ALLOCATION ($1,000 Minimum Increments)")
    print("-" * 60)
    print(f"Initial Capital Outlay : ${outlay:,.2f}")
    print(f"Notes Purchased        : {int(res['notes_bought']):,} notes (${res['par_discrete']:,.2f} par)")
    print(f"Actual Cash Spent      : ${res['actual_spent']:,.2f}")
    print(f"Unspent Cash Returned  : ${res['unspent_cash']:,.2f}")
    print(f"Total Coupons Received : ${res['total_coupons_discrete']:,.2f}")
    print(f"Principal at Maturity  : ${res['par_discrete']:,.2f}")
    print(f"TOTAL VALUE AT END     : ${res['total_payout_discrete']:,.2f}")

    print("\n" + "-" * 60)
    print("FRICTIONLESS ALLOCATION (100% Deployed / Fractional Par)")
    print("-" * 60)
    print(f"Theoretical Par Owned  : ${res['par_fractional']:,.2f}")
    print(f"Total Coupons Received : ${res['total_coupons_fractional']:,.2f}")
    print(f"TOTAL VALUE AT END     : ${res['total_payout_fractional']:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()