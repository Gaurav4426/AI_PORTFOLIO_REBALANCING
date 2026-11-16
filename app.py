
import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "historical_returns.csv"
MODEL_PATH = BASE_DIR / "models" / "portfolio_model.pkl"

INITIAL_BALANCE = 100000


def run_backtest(
    stock,
    bonds,
    gold,
    cash,
    etf,
    risk_aversion,
    max_drawdown,
    transaction_cost
):

    total = stock + bonds + gold + cash + etf

    if abs(total - 100) > 0.01:
        return (
            "Portfolio weights must add up to 100%.",
            "", "", "", "", None, None, None, ""
        )

    try:

        data = pd.read_csv(DATA_PATH)
        model = joblib.load(MODEL_PATH)

        assets = model["assets"]

        weights_dict = {
            "Stocks": stock / 100,
            "Bonds": bonds / 100,
            "Gold": gold / 100,
            "Cash": cash / 100,
            "ETF": etf / 100
        }

        weights = np.array([
            weights_dict[asset] for asset in assets
        ])

        returns = data[assets].astype(float)

        portfolio_returns = returns.dot(weights)

        turnover = np.sum(
            np.abs(weights - np.ones(len(weights)) / len(weights))
        ) / 2

        cost = turnover * transaction_cost / 100

        balance = INITIAL_BALANCE * (1 - cost)

        values = [balance]

        for daily_return in portfolio_returns:

            balance *= (1 + daily_return)

            values.append(balance)

        values = np.array(values)

        total_return = (
            (values[-1] / INITIAL_BALANCE) - 1
        ) * 100

        daily_returns = pd.Series(values).pct_change().dropna()

        volatility = (
            daily_returns.std() * np.sqrt(252) * 100
        )

        if daily_returns.std() != 0:

            sharpe = (
                (daily_returns.mean() - 0.05 / 252)
                / daily_returns.std()
            ) * np.sqrt(252)

        else:
            sharpe = 0

        running_max = np.maximum.accumulate(values)

        drawdown = (
            (values - running_max) / running_max
        ) * 100

        max_dd = abs(drawdown.min())

        dates = data["Date"].astype(str).tolist()

        dates = ["Initial"] + dates

        portfolio_chart = go.Figure()

        portfolio_chart.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name="Portfolio Value"
            )
        )

        portfolio_chart.update_layout(
            title="Historical Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            template="plotly_white"
        )

        allocation_chart = go.Figure()

        allocation_chart.add_trace(
            go.Pie(
                labels=assets,
                values=weights * 100,
                hole=0.4
            )
        )

        allocation_chart.update_layout(
            title="Selected Asset Allocation"
        )

        drawdown_chart = go.Figure()

        drawdown_chart.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode="lines",
                fill="tozeroy",
                name="Drawdown"
            )
        )

        drawdown_chart.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white"
        )

        status = "Historical backtesting completed successfully."

        if max_dd > max_drawdown:
            status += " Warning: Maximum drawdown exceeded your selected limit."

        model_weights = model["best_weights"]

        model_allocation = "\n".join([
            f"{asset}: {weight * 100:.2f}%"
            for asset, weight in zip(assets, model_weights)
        ])

        esg_message = (
            "ESG scores are not available in the current dataset. "
            "ESG constraints are not applied."
        )

        return (
            status,
            f"{total_return:.2f}%",
            f"{volatility:.2f}%",
            f"{sharpe:.2f}",
            f"{max_dd:.2f}%",
            portfolio_chart,
            allocation_chart,
            drawdown_chart,
            model_allocation + "\n\n" + esg_message
        )

    except Exception as e:

        return (
            f"Error: {str(e)}",
            "", "", "", "", None, None, None, ""
        )


with gr.Blocks(
    title="AI Portfolio Rebalancing",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # AI Portfolio Rebalancing

        ### Risk and ESG Constrained Portfolio Analysis

        An AI-based educational framework for historical
        portfolio analysis and asset allocation.
        """
    )

    with gr.Tab("Portfolio Setup"):

        gr.Markdown("## Asset Allocation (%)")

        with gr.Row():

            stock = gr.Slider(
                0, 100, value=30, step=1,
                label="Stocks"
            )

            bonds = gr.Slider(
                0, 100, value=25, step=1,
                label="Bonds"
            )

        with gr.Row():

            gold = gr.Slider(
                0, 100, value=15, step=1,
                label="Gold"
            )

            cash = gr.Slider(
                0, 100, value=10, step=1,
                label="Cash"
            )

            etf = gr.Slider(
                0, 100, value=20, step=1,
                label="ETF"
            )

        gr.Markdown(
            "Default allocation: 30% Stocks, 25% Bonds, "
            "15% Gold, 10% Cash and 20% ETF."
        )

    with gr.Tab("Risk Settings"):

        risk_aversion = gr.Slider(
            0, 1, value=0.5, step=0.1,
            label="Risk Aversion Preference"
        )

        max_drawdown = gr.Slider(
            0, 50, value=20, step=1,
            label="Maximum Drawdown Limit (%)"
        )

        transaction_cost = gr.Slider(
            0, 5, value=0.1, step=0.01,
            label="Transaction Cost (%)"
        )

        gr.Markdown(
            "Risk preference is displayed as a setting. "
            "The current backtest uses historical returns "
            "and reports whether the drawdown limit is exceeded."
        )

    with gr.Tab("Backtest"):

        gr.Markdown("## Historical Performance Analysis")

        run_button = gr.Button(
            "RUN BACKTEST",
            variant="primary"
        )

        status = gr.Textbox(label="Backtest Status")

        with gr.Row():

            return_output = gr.Textbox(
                label="Total Return"
            )

            risk_output = gr.Textbox(
                label="Annual Volatility"
            )

            sharpe_output = gr.Textbox(
                label="Sharpe Ratio"
            )

            drawdown_output = gr.Textbox(
                label="Maximum Drawdown"
            )

        portfolio_chart = gr.Plot(
            label="Portfolio Performance"
        )

        allocation_chart = gr.Plot(
            label="Asset Allocation"
        )

        drawdown_chart = gr.Plot(
            label="Drawdown Analysis"
        )

        model_output = gr.Textbox(
            label="Trained Model Allocation and ESG Data Status",
            lines=8
        )

        run_button.click(
            fn=run_backtest,
            inputs=[
                stock,
                bonds,
                gold,
                cash,
                etf,
                risk_aversion,
                max_drawdown,
                transaction_cost
            ],
            outputs=[
                status,
                return_output,
                risk_output,
                sharpe_output,
                drawdown_output,
                portfolio_chart,
                allocation_chart,
                drawdown_chart,
                model_output
            ]
        )


port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
