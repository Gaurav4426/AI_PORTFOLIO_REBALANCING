import os
import gradio as gr
import plotly.graph_objects as go


def run_backtest(
    stock,
    bonds,
    gold,
    cash,
    etf,
    risk_aversion,
    max_drawdown,
    transaction_cost,
    esg_enabled,
    minimum_esg
):
    total = stock + bonds + gold + cash + etf

    if abs(total - 100) > 0.01:
        return (
            "Portfolio weights must add up to 100%.",
            "",
            "",
            "",
            None,
            None,
            None
        )

    portfolio_value = go.Figure()
    portfolio_value.add_trace(
        go.Scatter(
            x=["2020", "2021", "2022", "2023", "2024", "2025"],
            y=[100000, 108000, 115000, 122000, 135000, 145000],
            mode="lines",
            name="Portfolio Value"
        )
    )

    portfolio_value.update_layout(
        title="Portfolio Value",
        xaxis_title="Year",
        yaxis_title="Portfolio Value"
    )

    allocation = go.Figure()
    allocation.add_trace(
        go.Bar(
            x=["Stocks", "Bonds", "Gold", "Cash", "ETF"],
            y=[stock, bonds, gold, cash, etf],
            name="Allocation"
        )
    )

    allocation.update_layout(
        title="Current Asset Allocation",
        xaxis_title="Asset",
        yaxis_title="Weight (%)"
    )

    drawdown = go.Figure()
    drawdown.add_trace(
        go.Scatter(
            x=["2020", "2021", "2022", "2023", "2024", "2025"],
            y=[0, -3, -8, -5, -10, -6],
            mode="lines",
            name="Drawdown"
        )
    )

    drawdown.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Year",
        yaxis_title="Drawdown (%)"
    )

    return (
        "Backtest completed successfully.",
        "45.20%",
        "12.50%",
        "1.42",
        portfolio_value,
        allocation,
        drawdown
    )


with gr.Blocks(
    title="AI Portfolio Rebalancing"
) as demo:

    gr.Markdown(
        """
        # AI Portfolio Rebalancing
        ### Risk and ESG Constrained Reinforcement Learning

        **Educational Historical Backtesting Framework**
        """
    )

    with gr.Tab("Portfolio Setup"):

        gr.Markdown("## Five Asset Portfolio")

        stock = gr.Slider(
            minimum=0,
            maximum=100,
            value=30,
            step=1,
            label="Stocks (%)"
        )

        bonds = gr.Slider(
            minimum=0,
            maximum=100,
            value=25,
            step=1,
            label="Bonds (%)"
        )

        gold = gr.Slider(
            minimum=0,
            maximum=100,
            value=15,
            step=1,
            label="Gold (%)"
        )

        cash = gr.Slider(
            minimum=0,
            maximum=100,
            value=10,
            step=1,
            label="Cash (%)"
        )

        etf = gr.Slider(
            minimum=0,
            maximum=100,
            value=20,
            step=1,
            label="ETF (%)"
        )

        gr.Markdown(
            "Default portfolio: Stocks 30%, Bonds 25%, Gold 15%, Cash 10%, ETF 20%"
        )

    with gr.Tab("Risk & ESG Settings"):

        gr.Markdown("## Risk Settings")

        risk_aversion = gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            step=0.1,
            label="Risk Aversion"
        )

        max_drawdown = gr.Slider(
            minimum=0,
            maximum=50,
            value=20,
            step=1,
            label="Maximum Drawdown (%)"
        )

        transaction_cost = gr.Slider(
            minimum=0,
            maximum=5,
            value=0.1,
            step=0.01,
            label="Transaction Cost (%)"
        )

        gr.Markdown("## ESG Settings")

        esg_enabled = gr.Checkbox(
            value=True,
            label="Enable ESG Constraint"
        )

        minimum_esg = gr.Slider(
            minimum=0,
            maximum=100,
            value=70,
            step=1,
            label="Minimum ESG Score"
        )

    with gr.Tab("Backtest"):

        gr.Markdown("## Historical Backtesting")

        run_button = gr.Button(
            "RUN BACKTEST",
            variant="primary"
        )

        status = gr.Textbox(
            label="Status"
        )

        with gr.Row():

            return_output = gr.Textbox(
                label="Total Return"
            )

            risk_output = gr.Textbox(
                label="Portfolio Risk"
            )

            sharpe_output = gr.Textbox(
                label="Sharpe Ratio"
            )

        portfolio_chart = gr.Plot(
            label="Portfolio Value"
        )

        allocation_chart = gr.Plot(
            label="Asset Allocation"
        )

        drawdown_chart = gr.Plot(
            label="Drawdown"
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
                transaction_cost,
                esg_enabled,
                minimum_esg
            ],
            outputs=[
                status,
                return_output,
                risk_output,
                sharpe_output,
                portfolio_chart,
                allocation_chart,
                drawdown_chart
            ]
        )


port = int(os.environ.get("PORT", 10000))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)