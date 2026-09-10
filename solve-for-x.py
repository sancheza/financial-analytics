#!/usr/bin/env python3
"""
Formula Solver
---------------
A command-line tool to solve percentage/proportion formulas of the form
"Rate% of X = Result", for whichever of the three values is unknown.

Modes:
    - Interactive TUI (default, no args): a full-screen form where Tab/Shift+Tab
      or the mouse move between fields, arrow keys or a click pick which field
      is unknown, and Enter/click Solve computes the answer.
    - Direct CLI mode (-f/--formula): solves one formula and prints the result,
      for scripting.
"""

import argparse
import re
import sys

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Checkbox, Footer, Header, Input, RadioButton, RadioSet, Static

FIELD_LABELS = {"Rate": "Percent (%)", "X": "Whole", "Result": "Part"}
FIELD_ORDER = ("Rate", "X", "Result")

HELP_TEXT = {
    "Rate": "[b]Solving for Rate[/b] — finds the percentage.\n"
            "Example: What interest rate earns $8,400 on a $200,000 principal?",
    "X": "[b]Solving for Whole[/b] — finds the base amount.\n"
         "Example: What principal is needed to earn $8,400 at 4.2%?",
    "Result": "[b]Solving for Part[/b] — finds the portion of the whole.\n"
              "Example: What annual income does $200,000 at 4.2% produce?",
}

TARGET_ALIASES = {
    "x": "X",
    "variable": "X",
    "base": "X",
    "whole": "X",
    "rate": "Rate",
    "percentage": "Rate",
    "percent": "Rate",
    "%": "Rate",
    "result": "Result",
    "total": "Result",
    "part": "Result",
    "=": "Result",
}


def target_type(value):
    """argparse type: normalizes -t/--target to one of 'X', 'Rate', 'Result'."""
    canonical = TARGET_ALIASES.get(value.strip().lower())
    if canonical is None:
        raise argparse.ArgumentTypeError(
            f"invalid target {value!r} (choose from X, Rate, Result)"
        )
    return canonical


def _to_float(value, field_name):
    try:
        s = str(value).strip().replace(",", "")
        if field_name == "Percent":
            s = s.rstrip("%")
        return float(s)
    except (TypeError, ValueError):
        raise ValueError(f"'{value}' is not a valid number for {field_name}.")


def solve_percentage_equation(rate, value_x, result, target_unknown, verbose=0):
    """Solves 'Percent% of Whole = Part' for whichever of the three is unknown.

    target_unknown selects which value to solve for ('X', 'Rate', or 'Result',
    case-insensitive; see TARGET_ALIASES for accepted synonyms, including
    'whole'/'percent'/'part'). The other two arguments must be provided.
    Raises ValueError on missing/invalid input.
    """
    steps = []
    target = TARGET_ALIASES.get(target_unknown.strip().lower())
    if target is None:
        raise ValueError(f"Unknown target variable: {target_unknown}")

    if target == "X":
        if rate is None or result is None:
            raise ValueError("Percent and Part are required to solve for Whole.")
        r_val = _to_float(rate, "Percent") / 100.0
        if r_val == 0:
            raise ValueError("Percent cannot be zero when solving for Whole.")
        res_val = _to_float(result, "Part")
        ans = res_val / r_val
        if verbose > 0:
            steps.append(f"1. Convert percentage to decimal: {rate}% -> {r_val}")
            steps.append(f"2. Formula: {r_val} * Whole = {res_val}")
            steps.append(f"3. Isolate Whole: Whole = {res_val} / {r_val}")
        return ans, steps

    if target == "Rate":
        if value_x is None or result is None:
            raise ValueError("Whole and Part are required to solve for Percent.")
        x_val = _to_float(value_x, "Whole")
        if x_val == 0:
            raise ValueError("Whole cannot be zero when solving for Percent.")
        res_val = _to_float(result, "Part")
        ans = (res_val / x_val) * 100.0
        if verbose > 0:
            steps.append(f"1. Formula: (Percent / 100) * {x_val} = {res_val}")
            steps.append(f"2. Isolate Percent: Percent = ({res_val} / {x_val}) * 100")
        return ans, steps

    # target == "Result"
    if rate is None or value_x is None:
        raise ValueError("Percent and Whole are required to solve for Part.")
    r_val = _to_float(rate, "Percent") / 100.0
    x_val = _to_float(value_x, "Whole")
    ans = r_val * x_val
    if verbose > 0:
        steps.append(f"1. Convert percentage to decimal: {rate}% -> {r_val}")
        steps.append(f"2. Multiply: Part = {r_val} * {x_val}")
    return ans, steps


def _numbers_to_operands(target, numbers):
    """Maps the two numbers found in a free-text formula onto (rate, x, result),
    assuming the standard 'Rate% of X = Result' left-to-right order."""
    n1, n2 = numbers[0], numbers[1]
    if target == "X":
        return n1, None, n2
    if target == "Rate":
        return None, n1, n2
    return n1, n2, None  # target == "Result"


PLACEHOLDERS = {"Rate": "e.g. 5%", "X": "e.g. 1000000", "Result": "e.g. 50000"}


class SolveForXApp(App):
    """Full-screen form: Tab/Shift+Tab or mouse move between fields, arrow keys
    or a click choose which field is unknown, Enter/click Solve computes it."""

    TITLE = "Percentage Solver"

    CSS = """
    Screen { align: center middle; }
    #panel {
        width: 60;
        max-height: 100%;
        border: round $foreground 30%;
        padding: 0 2;
    }
    #target-select { margin: 1 0; }
    Input { margin-bottom: 1; }
    #verbose { margin-bottom: 1; }
    #solve-btn { width: 100%; }
    #help-text {
        height: auto;
        min-height: 1;
        margin-top: 1;
        color: $text-muted;
        background: $surface-darken-1;
        border: round $foreground 30%;
        text-style: italic;
        padding: 0 2;
    }
    #output {
        height: auto;
        min-height: 3;
        border: round $foreground 30%;
        padding: 0 2;
        margin-top: 1;
        content-align: center middle;
        text-align: center;
    }
    #output.error { color: $text-error; text-style: bold; }
    #output.success { color: $text-success; text-style: bold; }
    """

    BINDINGS = [("q", "quit", "Quit")]

    unknown = "X"

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="panel"):
            with RadioSet(id="target-select", compact=True):
                yield RadioButton("Percent (%)  —  the rate or percentage", id="opt-Rate")
                yield RadioButton("Whole  —  the base or principal amount", id="opt-X", value=True)
                yield RadioButton("Part  —  the result or portion", id="opt-Result")
            for name in FIELD_ORDER:
                yield Input(id=f"input-{name}")
            yield Checkbox("Show step-by-step breakdown", id="verbose", compact=True)
            yield Button("Solve", id="solve-btn", variant="primary", compact=True)
            yield Static("", id="output")
            yield Static("", id="help-text")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#panel").border_title = "Percent% of Whole = Part"
        self.query_one("#output").border_title = "Result"
        for name in FIELD_ORDER:
            self.query_one(f"#input-{name}", Input).border_title = FIELD_LABELS[name]
        self._set_unknown("X")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._set_unknown(event.pressed.id.split("-", 1)[1])

    def _set_unknown(self, target: str) -> None:
        self.unknown = target
        self.query_one("#help-text").update(HELP_TEXT[target])
        for name in FIELD_ORDER:
            field = self.query_one(f"#input-{name}", Input)
            if name == target:
                field.value = ""
                field.disabled = True
                field.placeholder = "(solving for this)"
            else:
                field.disabled = False
                field.placeholder = PLACEHOLDERS[name]

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._solve()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "solve-btn":
            self._solve()

    def _solve(self) -> None:
        output = self.query_one("#output", Static)
        values = {
            name: self.query_one(f"#input-{name}", Input).value.strip()
            for name in FIELD_ORDER
            if name != self.unknown
        }
        verbose = self.query_one("#verbose", Checkbox).value

        try:
            ans, steps = solve_percentage_equation(
                values.get("Rate"),
                values.get("X"),
                values.get("Result"),
                self.unknown,
                verbose=1 if verbose else 0,
            )
        except ValueError as e:
            output.update(f"Error: {e}")
            output.set_classes(["error"])
            return

        lines = list(steps) if verbose else []
        if lines:
            lines.append("")
        lines.append(f"{FIELD_LABELS[self.unknown]} = {ans:,.4f}")
        output.update("\n".join(lines))
        output.set_classes(["success"])


def run_tui() -> None:
    if not sys.stdin.isatty():
        print(
            "Interactive mode requires a terminal. Use -f/--formula for "
            "non-interactive use, e.g.: solve-for-x.py -f '5% of X = 215000'"
        )
        sys.exit(1)
    SolveForXApp().run()


def main():
    parser = argparse.ArgumentParser(
        description="Solve percentage and proportion formulas dynamically."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show step-by-step algebraic breakdown",
    )
    parser.add_argument(
        "-f",
        "--formula",
        type=str,
        help="Formula string to evaluate directly (e.g. '5%% of X = 215000')",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=target_type,
        default="X",
        metavar="{X,Rate,Result}",
        help="Target variable to solve for: X, Rate, or Result",
    )

    args = parser.parse_args()

    if not args.formula:
        run_tui()
        return

    numbers = re.findall(r"([0-9.]+)", args.formula)
    if len(numbers) < 2:
        print("Invalid formula format provided. Need at least two numbers, e.g. '5% of X = 215000'.")
        sys.exit(1)

    rate, x_val, result = _numbers_to_operands(args.target, numbers)

    try:
        ans, steps = solve_percentage_equation(
            rate, x_val, result, args.target, verbose=1 if args.verbose else 0
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.verbose:
        for s in steps:
            print(s)
    print(f"Solution: {FIELD_LABELS[args.target]} = {ans:,.4f}")


if __name__ == "__main__":
    main()
