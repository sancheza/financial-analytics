import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

tickers = ["AMAT", "AOS", "DHI", "LULU", "NVR", "ON", "PHM", "SNA"]

end = datetime.today()
start = end - timedelta(days=30)

# Download historical data
data = yf.download(tickers, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

# Build summary
results = []
for ticker in tickers:
    highs = data['High'][ticker]
    lows = data['Low'][ticker]
    results.append({
        "Ticker": ticker,
        "30D Low": round(lows.min(), 2),
        "30D High": round(highs.max(), 2)
    })

# Output results
df = pd.DataFrame(results)
print(df.to_string(index=False))

