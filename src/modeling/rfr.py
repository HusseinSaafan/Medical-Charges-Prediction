from sklearn.ensemble import RandomForestRegressor

from src.utils.helpers import (
    build_train_metrics_payload,
    compute_regression_metrics,
    load_training_data,
    run_stratified_grid_search,
    save_train_metrics,
)


def run_random_forest() -> dict:
    X_train, y_train = load_training_data()

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid, _, _ = run_stratified_grid_search(
        model=model,
        param_grid=param_grid,
        model_name="random_forest_regressor",
    )

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_train)

    train_metrics = compute_regression_metrics(y_train, y_pred, n_features=X_train.shape[1])
    amse = -float(grid.best_score_)

    payload = build_train_metrics_payload(
        model_name="RandomForestRegressor",
        num_samples=len(X_train),
        num_features=X_train.shape[1],
        train_metrics=train_metrics,
        amse=amse,
        best_params=grid.best_params_,
    )

    save_train_metrics(payload, "random_forest_regressor_train_metrics.txt")

    return {
        "model_key": "rfr",
        "model_name": "RandomForestRegressor",
        "estimator": best_model,
        "AMSE": amse,
        "best_params": grid.best_params_,
        "train_metrics": payload,
    }


if __name__ == "__main__":
    run_random_forest()
