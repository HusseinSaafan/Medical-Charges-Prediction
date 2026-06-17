from math import sqrt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.utils.config import PROJECT_ROOT, log

TRAIN_DATA_PATH = PROJECT_ROOT / "database" / "train_encoded.csv"
TEST_DATA_PATH = PROJECT_ROOT / "database" / "test_encoded.csv"
TARGET_COLUMN = "charges"
CV_RESULTS_DIR = PROJECT_ROOT / "artifacts" / "cv_results"
TRAIN_RESULTS_DIR = PROJECT_ROOT / "src" / "evaluation" / "model_train_results"


def load_dataset(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from {file_path}")
    return df


def load_training_data(file_path: Path = TRAIN_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    log("Loading training data from:", file_path)
    df = load_dataset(file_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    log("Training data loaded for CV. X shape:", X.shape, "y shape:", y.shape)
    return X, y


def load_test_data(file_path: Path = TEST_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    log("Loading test data from:", file_path)
    df = load_dataset(file_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    log("Test data loaded. X shape:", X.shape, "y shape:", y.shape)
    return X, y


def _build_strat_labels(y: pd.Series, n_bins: int = 5, n_splits: int = 5) -> pd.Series:
    n_unique = int(y.nunique())
    bins = min(n_bins, n_unique)

    if bins < 2:
        raise ValueError("Target has fewer than 2 unique values; cannot stratify")

    labels = pd.qcut(y, q=bins, duplicates="drop")

    while labels.nunique() > 2 and labels.value_counts().min() < n_splits:
        bins -= 1
        labels = pd.qcut(y, q=bins, duplicates="drop")

    return pd.Series(labels.cat.codes, index=y.index)


def compute_regression_metrics(y_true: pd.Series, y_pred: pd.Series, n_features: int) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)
    if n > (n_features + 1):
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    else:
        adjusted_r2 = float("nan")

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Adjusted_R2": float(adjusted_r2),
    }


def save_train_metrics(metrics: Dict[str, Any], output_filename: str, output_dir: Path = TRAIN_RESULTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename

    lines = [f"{key}: {value}" for key, value in metrics.items()]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("Saved model training results to:", output_file)
    return output_file


def display_cv_results(cv_results_df: pd.DataFrame, model_name: str, top_n: int = 10) -> pd.DataFrame:
    keep_cols = [
        "rank_test_neg_mse",
        "mean_test_neg_mse",
        "std_test_neg_mse",
        "mean_test_r2",
        "std_test_r2",
        "mean_test_neg_mae",
        "std_test_neg_mae",
        "params",
    ]

    available_cols = [col for col in keep_cols if col in cv_results_df.columns]
    preview_df = cv_results_df[available_cols].sort_values(by="rank_test_neg_mse").head(top_n)

    log(f"Top {top_n} CV rows for {model_name}:")
    for _, row in preview_df.iterrows():
        log(
            "rank:",
            int(row["rank_test_neg_mse"]),
            "neg_mse:",
            float(row["mean_test_neg_mse"]),
            "r2:",
            float(row["mean_test_r2"]),
            "neg_mae:",
            float(row["mean_test_neg_mae"]),
            "params:",
            row["params"],
        )

    return preview_df


def save_cv_results_html(
    cv_results_df: pd.DataFrame,
    model_name: str,
    n_splits: int = 5,
    n_samples: int = 0,
    n_features: int = 0,
    output_dir: Path = CV_RESULTS_DIR,
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name.lower()}_cv_results.html"

    n_train_fold = int(n_samples * (n_splits - 1) / n_splits) if n_samples > 0 else 1
    best_row = cv_results_df[cv_results_df["rank_test_neg_mse"] == 1].iloc[0]

    fold_labels = [f"Fold {i + 1}" for i in range(n_splits)]

    mse_vals = [-best_row[f"split{i}_test_neg_mse"] for i in range(n_splits)]
    rmse_vals = [sqrt(v) for v in mse_vals]
    mae_vals = [-best_row[f"split{i}_test_neg_mae"] for i in range(n_splits)]
    r2_vals = [float(best_row[f"split{i}_test_r2"]) for i in range(n_splits)]

    def adj_r2(r2: float) -> float:
        if n_samples < 2 or n_features < 1 or (n_train_fold - n_features - 1) <= 0:
            return float("nan")
        return 1 - (1 - r2) * (n_train_fold - 1) / (n_train_fold - n_features - 1)

    adj_r2_vals = [adj_r2(v) for v in r2_vals]

    metrics = [
        ("MSE", mse_vals, -float(best_row["mean_test_neg_mse"]), "#EF553B"),
        ("RMSE", rmse_vals, sqrt(-float(best_row["mean_test_neg_mse"])), "#AB63FA"),
        ("MAE", mae_vals, -float(best_row["mean_test_neg_mae"]), "#FF7F0E"),
        ("R2", r2_vals, float(best_row["mean_test_r2"]), "#00CC96"),
        ("Adjusted_R2", adj_r2_vals, adj_r2(float(best_row["mean_test_r2"])), "#19D3F3"),
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[m[0] for m in metrics],
        shared_yaxes=False,
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]

    for (metric_name, values, mean_val, color), (row, col) in zip(metrics, positions):
        fig.add_trace(
            go.Bar(
                x=fold_labels,
                y=values,
                name=metric_name,
                marker_color=color,
                opacity=0.75,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=fold_labels,
                y=[mean_val] * n_splits,
                mode="lines",
                name=f"Mean {metric_name}",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=True,
            ),
            row=row,
            col=col,
        )

    best_params = best_row["params"]
    fig.update_layout(
        title=dict(
            text=(
                f"{model_name} — Stratified {n_splits}-Fold CV Scores<br>"
                f"<sub>Best params: {best_params}</sub>"
            ),
            font=dict(size=14),
        ),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
        template="plotly_white",
    )

    fig.write_html(str(output_file), include_plotlyjs="cdn")
    log("Saved CV results chart to:", output_file)
    return output_file


def run_stratified_grid_search(
    model: Any,
    param_grid: Dict[str, Any],
    model_name: str,
    X: Optional[pd.DataFrame] = None,
    y: Optional[pd.Series] = None,
    n_splits: int = 5,
    n_bins: int = 5,
    random_state: int = 42,
) -> Tuple[GridSearchCV, pd.DataFrame, Path]:
    if X is None or y is None:
        X, y = load_training_data()

    strat_labels = _build_strat_labels(y, n_bins=n_bins, n_splits=n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "neg_mse": "neg_mean_squared_error",
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="neg_mse",
        cv=cv.split(X, strat_labels),
        n_jobs=1,
        verbose=0,
        return_train_score=True,
    )

    log(f"Running stratified CV + grid search for {model_name}")
    grid.fit(X, y)

    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_df = cv_results_df.sort_values(by="rank_test_neg_mse", ascending=True)

    display_cv_results(cv_results_df, model_name=model_name, top_n=10)
    html_path = save_cv_results_html(
        cv_results_df,
        model_name=model_name,
        n_splits=n_splits,
        n_samples=int(X.shape[0]),
        n_features=int(X.shape[1]),
    )

    log(f"Best params for {model_name}:", grid.best_params_)
    log(f"Best CV neg_mse for {model_name}:", float(grid.best_score_))

    return grid, cv_results_df, html_path


def build_train_metrics_payload(
    model_name: str,
    num_samples: int,
    num_features: int,
    train_metrics: Dict[str, float],
    amse: float,
    best_params: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": "train_encoded.csv",
        "model": model_name,
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "num_samples": int(num_samples),
        "num_features": int(num_features),
        "MSE": float(train_metrics["MSE"]),
        "RMSE": float(train_metrics["RMSE"]),
        "MAE": float(train_metrics["MAE"]),
        "R2": float(train_metrics["R2"]),
        "AMSE": float(amse),
        "best_params": best_params,
    }
