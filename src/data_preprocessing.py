import yfinance as yf
import pandas as pd
import os


def download_data(start_date="2020-01-01", end_date="2025-12-31"):

    assets = {
        "Stocks": "SPY",
        "Bonds": "BND",
        "Gold": "GLD",
        "ETF": "QQQ"
    }

    data = {}

    for name, ticker in assets.items():

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        if not df.empty:
            close = df["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            data[name] = close

    prices = pd.DataFrame(data)

    prices["Cash"] = 1.0

    prices = prices.dropna()

    returns = prices.pct_change().dropna()

    return prices, returns


if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    prices, returns = download_data()

    prices.to_csv("data/historical_prices.csv")
    returns.to_csv("data/historical_returns.csv")

    print("Historical data saved successfully!")