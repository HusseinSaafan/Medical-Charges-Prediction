import os
from pathlib import Path

CONDA_LIBOMP = Path('/opt/anaconda3/lib')
if (CONDA_LIBOMP / 'libomp.dylib').exists():
    current = os.environ.get('DYLD_LIBRARY_PATH', '')
    merged = [str(CONDA_LIBOMP)]
    if current:
        merged.extend(p for p in current.split(':') if p)
    os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = ':'.join(dict.fromkeys(merged))
    os.environ['DYLD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(merged))

from xgboost import XGBRegressor

from src.utils.helpers import (
    build_train_metrics_payload,
    compute_regression_metrics,
    load_training_data,
    run_stratified_grid_search,
    save_train_metrics,
)


def run_xgboost_regressor() -> dict:
    X_train, y_train = load_training_data()

    model = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
    )
    param_grid = {
        'n_estimators': [200, 400],
        'learning_rate': [0.03, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    grid, _, _ = run_stratified_grid_search(
        model=model,
        param_grid=param_grid,
        model_name='xgboost_regressor',
    )

    best_model = grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_train)

    train_metrics = compute_regression_metrics(y_train, y_pred, n_features=X_train.shape[1])
    amse = -float(grid.best_score_)

    payload = build_train_metrics_payload(
        model_name='XGBoostRegressor',
        num_samples=len(X_train),
        num_features=X_train.shape[1],
        train_metrics=train_metrics,
        amse=amse,
        best_params=grid.best_params_,
    )

    save_train_metrics(payload, 'xgboost_regressor_train_metrics.txt')

    return {
        'model_key': 'xgbr',
        'model_name': 'XGBoostRegressor',
        'estimator': best_model,
        'AMSE': amse,
        'best_params': grid.best_params_,
        'train_metrics': payload,
    }


if __name__ == '__main__':
    run_xgboost_regressor()
