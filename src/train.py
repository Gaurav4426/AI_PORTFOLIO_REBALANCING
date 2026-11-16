
import os
import numpy as np
import joblib
from src.environment import PortfolioEnvironment

env = PortfolioEnvironment()

episodes = 100
best_reward = -float("inf")
best_weights = None

for episode in range(episodes):

    state = env.reset()
    total_reward = 0
    done = False

    while not done:

        action = np.random.dirichlet(np.ones(env.num_assets))

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

    if total_reward > best_reward:
        best_reward = total_reward
        best_weights = env.weights.copy()

    if (episode + 1) % 10 == 0:
        print("Episode:", episode + 1, "Reward:", round(total_reward, 4))

os.makedirs("models", exist_ok=True)

model = {
    "best_weights": best_weights,
    "best_reward": best_reward,
    "assets": env.assets
}

joblib.dump(model, "models/portfolio_model.pkl")

print("\nTraining completed successfully!")
print("Best Reward:", best_reward)
print("Best Weights:", best_weights)
print("Model saved successfully!")
