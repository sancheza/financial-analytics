# Finance scripts repo

### getBookValue

This is a Google Sheets function that populates book value in a worksheet for a set of tickers.

#### Usage
Create a Google worksheet with stock tickers in one column and a trigger cell used for activating the function in another. In the demo below, tickers are housed in column A and the trigger cell is N2. The trigger cell is useful if you are using a free API with limits. If not, feel free to remove it.

You will need to update the function with your API key. You can obtain it at the link listed below under Dependencies.

The function will populate the cell in which it is called as well as the two to the right of it. See cells I2 through K2 in the demo below.


```Dependencies: Google Sheets, Financial Modeling Prep (FMP) free API key```[(get it here)](https://financialmodelingprep.com)

![](https://github.com/sancheza/Finance-scripts/blob/main/getBookValueFMP-demo.gif)


### get-dividends

This script obtains dividend yield for a list of stocks defined in ticker.txt and outputs in the file dividends.txt. This version leverages the Yahoo Finance API to pull dividend data.

```Dependencies: Python 3, yfinance module```

![](https://github.com/sancheza/Finance-scripts/blob/main/get-dividends-demo.gif)



