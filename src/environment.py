
import numpy as np
import pandas as pd


class PortfolioEnvironment:

    def __init__(self, data_path="data/historical_returns.csv", initial_balance=100000):

        self.data = pd.read_csv(data_path)
        self.initial_balance = initial_balance
        self.balance = initial_balance

        self.assets = self.data.select_dtypes(include=[np.number]).columns.tolist()

        self.num_assets = len(self.assets)

        self.current_step = 0

        self.weights = np.ones(self.num_assets) / self.num_assets

    def reset(self):

        self.current_step = 0
        self.balance = self.initial_balance
        self.weights = np.ones(self.num_assets) / self.num_assets

        return self.get_state()

    def get_state(self):

        return np.concatenate([
            self.weights,
            [self.balance / self.initial_balance]
        ])

    def step(self, action):

        action = np.array(action)

        action = np.maximum(action, 0)

        if np.sum(action) == 0:
            action = np.ones(self.num_assets) / self.num_assets
        else:
            action = action / np.sum(action)

        self.weights = action

        returns = self.data[self.assets].iloc[self.current_step].values

        portfolio_return = np.dot(self.weights, returns)

        self.balance *= (1 + portfolio_return)

        reward = portfolio_return

        self.current_step += 1

        done = self.current_step >= len(self.data)

        if done:
            next_state = self.get_state()
        else:
            next_state = self.get_state()

        return next_state, reward, done, {
            "balance": self.balance,
            "portfolio_return": portfolio_return
        }
