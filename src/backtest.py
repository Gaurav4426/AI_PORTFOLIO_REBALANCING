
import os
import joblib
import numpy as np
import pandas as pd

from src.environment import PortfolioEnvironment

env = PortfolioEnvironment()

model = joblib.load("models/portfolio_model.pkl")

weights = model["best_weights"]

data = env.data[env.assets]

initial_balance = 100000
balance = initial_balance

portfolio_values = []

for i in range(len(data)):

    returns = data.iloc[i].values

    portfolio_return = np.dot(weights, returns)

    balance = balance * (1 + portfolio_return)

    portfolio_values.append(balance)

total_return = ((balance - initial_balance) / initial_balance) * 100

os.makedirs("results", exist_ok=True)

results = pd.DataFrame({
    "Portfolio Value": portfolio_values
})

results.to_csv("results/backtest_results.csv", index=False)

print("Backtesting completed successfully!")
print("Initial Balance:", initial_balance)
print("Final Balance:", round(balance, 2))
print("Total Return:", round(total_return, 2), "%")
print("Results saved in results/backtest_results.csv")
