# initial exploratory data analysis for the medical charges dataset
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import PROJECT_ROOT, log

FIGURES_DIR = PROJECT_ROOT / "figures"

# load and analyze the dataset keeping logging the results
def load_and_analyze_dataset(file_path):
    log("Loading dataset from:", file_path)
    df = pd.read_csv(file_path)
    log("Dataset loaded successfully. Shape:", df.shape)

    log("First 5 rows of the dataset:")
    log(df.head())

    log("Dataset info:")
    info_buffer = StringIO()
    df.info(buf=info_buffer)
    log(info_buffer.getvalue())

    log("Summary statistics:")
    log(df.describe())

    return df

# diplay nan values and distinct values for each column
def display_nan_and_distinct_values(df):
    log("Checking for NaN values in each column:")
    log(df.isna().sum())

    log("Checking for distinct values in each column:")
    for column in df.columns:
        distinct_values = df[column].unique()
        log(f"Column '{column}' has {len(distinct_values)} distinct values: {distinct_values}")

# visualize the distribution of all features using KDE plots and save to the figures directory
def visualize_feat_distributions(df):
    from scipy.stats import gaussian_kde

    log("Visualizing the distribution of all features using KDE plots")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in numeric_cols:
        # Create histogram
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"Distribution of {col}",
            labels={col: col},
        )

        # Compute KDE
        kde = gaussian_kde(df[col].dropna())
        x_range = np.linspace(df[col].min(), df[col].max(), 200)
        kde_values = kde(x_range)

        # Scale KDE to histogram scale
        bin_width = (df[col].max() - df[col].min()) / 30
        kde_scaled = kde_values * len(df[col]) * bin_width

        # Add KDE trace
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_scaled,
                mode="lines",
                name="KDE",
                line=dict(color="red", width=2),
            )
        )

        output_file = FIGURES_DIR / f"{col}_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_file, include_plotlyjs="cdn")
        log(f"Saved plot to: {output_file}")

if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "database" / "raw" / "insurance.csv"
    df = load_and_analyze_dataset(dataset_path)
    display_nan_and_distinct_values(df)
    visualize_feat_distributions(df)