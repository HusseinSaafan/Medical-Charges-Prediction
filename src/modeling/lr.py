import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression

from src.utils.config import PROJECT_ROOT, log
from src.utils.helpers import (
    build_train_metrics_payload,
    compute_regression_metrics,
    load_training_data,
    run_stratified_grid_search,
    save_train_metrics,
)

MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def _save_model_pickle(model, filename: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / filename
    with model_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)
    log("Saved model pickle to:", model_path)
    return model_path


def run_linear_regression() -> dict:
    X_train, y_train = load_training_data()

    model = LinearRegression()
    param_grid = {
        "fit_intercept": [True, False],
    }

    grid, _, _ = run_stratified_grid_search(
        model=model,
        param_grid=param_grid,
        model_name="linear_regression",
    )

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_train)

    train_metrics = compute_regression_metrics(y_train, y_pred, n_features=X_train.shape[1])
    amse = -float(grid.best_score_)

    payload = build_train_metrics_payload(
        model_name="LinearRegression",
        num_samples=len(X_train),
        num_features=X_train.shape[1],
        train_metrics=train_metrics,
        amse=amse,
        best_params=grid.best_params_,
    )

    save_train_metrics(payload, "linear_regression_train_metrics.txt")
    model_path = _save_model_pickle(best_model, "lr.pkl")

    return {
        "model_key": "lr",
        "model_name": "LinearRegression",
        "estimator": best_model,
        "AMSE": amse,
        "best_params": grid.best_params_,
        "train_metrics": payload,
        "model_path": model_path,
    }


if __name__ == "__main__":
    run_linear_regression()
