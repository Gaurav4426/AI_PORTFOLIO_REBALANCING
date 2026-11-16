
import numpy as np
import pandas as pd

results = pd.read_csv("results/backtest_results.csv")

portfolio_values = results["Portfolio Value"].values

returns = pd.Series(portfolio_values).pct_change().dropna()

total_return = (
    (portfolio_values[-1] - portfolio_values[0])
    / portfolio_values[0]
) * 100

volatility = returns.std() * np.sqrt(252) * 100

risk_free_rate = 0.05 / 252

sharpe_ratio = (
    (returns.mean() - risk_free_rate) / returns.std()
) * np.sqrt(252)

running_max = np.maximum.accumulate(portfolio_values)

drawdown = (portfolio_values - running_max) / running_max

max_drawdown = abs(drawdown.min()) * 100

print("\nPORTFOLIO PERFORMANCE METRICS")
print("--------------------------------")
print("Total Return:", round(total_return, 2), "%")
print("Annual Volatility:", round(volatility, 2), "%")
print("Sharpe Ratio:", round(sharpe_ratio, 2))
print("Maximum Drawdown:", round(max_drawdown, 2), "%")
