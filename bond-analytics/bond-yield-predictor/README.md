# Bond Yield Predictor

## Overview
The Bond Yield Predictor is a Python project designed to predict the expected yield or price of U.S. Treasury bonds for upcoming auctions. It utilizes historical yield data and auction results to generate predictions using various statistical methods.

## Features
- Fetches historical yield data from the FRED API.
- Scrapes auction schedules and results from TreasuryDirect.gov.
- Implements prediction logic using rolling averages, trend regressions, and statistical models like ARIMA.
- Caches data in JSON format for efficient access.
- Provides command-line interface for easy usage.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bond-yield-predictor.git
   cd bond-yield-predictor
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your FRED API key:
   - Create a `.env` file in the root directory and add your API key:
     ```
     FRED_API_KEY=your_api_key_here
     ```

## Usage
Run the script from the command line with the following options:

```
python src/main.py [options]
```

### Options
- `-h`, `--help`: Show usage information.
- `-v`, `--version`: Show script version.
- `--bondtype [10Y|20Y|30Y|etc.]`: Specify bond type. If omitted, predict all supported types.
- `--force-update`: Bypass cache and fetch fresh data.
- `--show-prediction`: Print the most recent prediction to console.

## Data Storage
- Cached data for each bond type is stored in `data/json/{bondtype}.json`.
- Backtest metrics are stored in `data/json/metrics_{bondtype}.json`.
- Logs of operations and errors are recorded in `data/log.txt`.

## Testing
To run the unit tests, execute:
```
pytest tests/
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- FRED API for historical yield data.
- TreasuryDirect.gov for auction schedules and results.