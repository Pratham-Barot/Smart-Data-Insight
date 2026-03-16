import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_histograms(df: pd.DataFrame, numeric_cols: list) -> list:
    """Generate histogram for every numeric column."""
    charts = []
    for col in numeric_cols:
        fig = px.histogram(
            df, x=col,
            title=f"Distribution of {col}",
            color_discrete_sequence=["#636EFA"],
            marginal="box"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        charts.append((col, fig))
    return charts


def generate_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Generate correlation heatmap for numeric columns."""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=True
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def generate_bar_charts(df: pd.DataFrame, categorical_cols: list) -> list:
    """Generate bar charts for categorical columns."""
    charts = []
    for col in categorical_cols:
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, "Count"]
        fig = px.bar(
            value_counts,
            x=col, y="Count",
            title=f"Value Counts — {col}",
            color=col,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        charts.append((col, fig))
    return charts


def generate_boxplots(df: pd.DataFrame, numeric_cols: list) -> list:
    """Generate boxplot for every numeric column to show outliers."""
    charts = []
    for col in numeric_cols:
        fig = px.box(
            df, y=col,
            title=f"Boxplot — {col}",
            color_discrete_sequence=["#EF553B"]
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        charts.append((col, fig))
    return charts


def generate_datetime_trend(df: pd.DataFrame, datetime_col: str,
                             numeric_cols: list):
    """Generate time series trend if datetime column exists."""
    if not datetime_col or not numeric_cols:
        return None
    try:
        df_temp = df.copy()
        df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col])
        df_temp = df_temp.sort_values(datetime_col)
        target_col = numeric_cols[0]
        fig = px.line(
            df_temp,
            x=datetime_col,
            y=target_col,
            title=f"Trend — {target_col} over {datetime_col}",
            color_discrete_sequence=["#00CC96"]
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def generate_scatter_matrix(df: pd.DataFrame, numeric_cols: list):
    """Generate scatter matrix for first 4 numeric columns."""
    cols = numeric_cols[:4]
    if len(cols) < 2:
        return None
    fig = px.scatter_matrix(
        df, dimensions=cols,
        title="Scatter Matrix",
        color_discrete_sequence=["#AB63FA"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def generate_pie_charts(df: pd.DataFrame, categorical_cols: list) -> list:
    """Generate pie charts for low cardinality categorical columns."""
    charts = []
    for col in categorical_cols:
        if df[col].nunique() <= 8:
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = [col, "Count"]
            fig = px.pie(
                value_counts,
                names=col,
                values="Count",
                title=f"Distribution — {col}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
            )
            charts.append((col, fig))
    return charts