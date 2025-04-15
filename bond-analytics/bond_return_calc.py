#!/usr/bin/env python3
import sys
import datetime

VERSION = "1.0.1"

def print_help():
    help_text = """\
bond_return_calc.py - Calculate total and annualized return for a bond

USAGE:
    bond_return_calc.py <coupon> <maturity_MM/DD/YYYY> <price>

DESCRIPTION:
    Calculates the total compounded and annualized return assuming 4.3% reinvestment.
    Accepts inputs in the same format as E*TRADE bond listings.

OPTIONS:
    -h              Show this help message
    -v              Show version information
"""
    print(help_text)

def print_version():
    print(f"bond_return_calc.py version {VERSION}")

def future_value_of_coupons(coupon: float, rate: float, years: float) -> float:
    return coupon * (((1 + rate) ** years - 1) / rate)

def calculate_returns(coupon: float, maturity: datetime.date, price: float, reinvest_rate: float = 0.043):
    today = datetime.date.today()
    years = (maturity - today).days / 365.25

    fv_coupons = future_value_of_coupons(coupon, reinvest_rate, years)
    fv_total = fv_coupons + 100  # principal returned at maturity

    total_return_pct = ((fv_total / price) - 1) * 100
    annual_return_pct = ((fv_total / price) ** (1 / years) - 1) * 100

    return fv_total, total_return_pct, annual_return_pct

def main():
    args = sys.argv[1:]

    if len(args) == 1 and args[0] == "-h":
        print_help()
        sys.exit(0)
    elif len(args) == 1 and args[0] == "-v":
        print_version()
        sys.exit(0)
    elif len(args) != 3:
        print_help()
        sys.exit(1)

    try:
        coupon = float(args[0])
        maturity = datetime.datetime.strptime(args[1], "%m/%d/%Y").date()
        price = float(args[2])

        fv, total_return, annual_return = calculate_returns(coupon, maturity, price)

        print(f"Input: {coupon}% coupon, matures {maturity}, price ${price}")
        print(f"Future Value (incl. reinvested coupons): ${fv:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annualized Return: {annual_return:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
