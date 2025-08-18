
import plotly.graph_objects as go
import pandas as pd

def plot_candles(df: pd.DataFrame, date_col: str, open_col: str, high_col: str, low_col: str, close_col: str, title: str):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df[date_col],
                open=df[open_col],
                high=df[high_col],
                low=df[low_col],
                close=df[close_col],
                name="Price"
            )
        ]
    )
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=500)
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def plot_line(df: pd.DataFrame, x: str, y_cols: list[str], names: list[str], title: str, show_volume: bool=False, volume_col: str="volume"):
    fig = go.Figure()
    for col, name in zip(y_cols, names):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=name))
    if show_volume and volume_col in df.columns:
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False)
        )
        fig.add_trace(
            go.Bar(x=df[x], y=df[volume_col], name="Volume", opacity=0.3, yaxis="y2")
        )
    fig.update_layout(title=title, height=400, xaxis_title="Date")
    return fig
