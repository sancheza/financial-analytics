# Financial Analytics Test Suite

This directory contains a centralized validation framework designed to ensure the integrity of your financial analytics scripts. It automates the verification of script execution, output accuracy, environment configuration, and file-system side effects.

---

## Prerequisites

Before running the suite, ensure you have the necessary dependencies installed:

* **pytest**: The core testing framework.
* **python-dotenv**: To manage your `.env` file (containing your `ALPHA_VANTAGE_KEY`).

```bash
pip install pytest python-dotenv
```

---

## Core Validations

The suite performs four primary levels of verification:

1.  **Process Integrity:** Confirms every script finishes with an exit code of 0.
2.  **Data Integrity:** Validates that script output matches specific financial data patterns (e.g., `Jan 23, 2026`, currency symbols, and percentage values).
3.  **Environment Integrity:** Checks for mandatory API keys in your environment before attempting network-dependent tests.
4.  **Side Effects:** Specifically verifies that scripts like `generate_auction_calendar.py` correctly create and populate `.ics` files.

---

## Usage Guide

All tests should be executed from the project root directory: `~/dev/financial-analytics/`.

### 1. Run All Tests
Execute the full suite, including long-running screener scripts.
```bash
pytest tests/test_financial_scripts.py -v
```

### 2. Fast Mode (Skip Slow Scripts)
Some scripts, such as `value_price_screener.py`, require significant time. Use the `-m` (marker) flag to exclude them.
```bash
pytest tests/test_financial_scripts.py -v -m "not slow"
```

### 3. Direct CLI Execution
The test file can be run directly to access its built-in help menu or trigger a run via Python.
```bash
python3 tests/test_financial_scripts.py --help
python3 tests/test_financial_scripts.py --run-now --fast
