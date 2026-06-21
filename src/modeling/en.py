import pickle
from pathlib import Path

from sklearn.linear_model import ElasticNet

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


def run_elastic_net() -> dict:
    X_train, y_train = load_training_data()

    model = ElasticNet(random_state=42, max_iter=10000)
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }

    grid, _, _ = run_stratified_grid_search(
        model=model,
        param_grid=param_grid,
        model_name="elastic_net",
    )

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_train)

    train_metrics = compute_regression_metrics(y_train, y_pred, n_features=X_train.shape[1])
    amse = -float(grid.best_score_)

    payload = build_train_metrics_payload(
        model_name="ElasticNet",
        num_samples=len(X_train),
        num_features=X_train.shape[1],
        train_metrics=train_metrics,
        amse=amse,
        best_params=grid.best_params_,
    )

    save_train_metrics(payload, "elastic_net_train_metrics.txt")
    model_path = _save_model_pickle(best_model, "en.pkl")

    return {
        "model_key": "en",
        "model_name": "ElasticNet",
        "estimator": best_model,
        "AMSE": amse,
        "best_params": grid.best_params_,
        "train_metrics": payload,
        "model_path": model_path,
    }


if __name__ == "__main__":
    run_elastic_net()
