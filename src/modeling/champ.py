import pickle
from pathlib import Path

import pandas as pd

from src.modeling.en import run_elastic_net
from src.modeling.lr import run_linear_regression
from src.modeling.rfr import run_random_forest
from src.modeling.xgbr import run_xgboost_regressor
from src.utils.config import PROJECT_ROOT, log
from src.utils.helpers import compute_regression_metrics, load_test_data

CHAMP_TEST_DIR = PROJECT_ROOT / "src" / "evaluation" / "champ_model_test"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"


def _save_model_pickle(model, filename: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / filename
    with model_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)
    log("Saved model pickle to:", model_path)
    return model_path


def _save_champ_test_html(champ_result: dict, test_metrics: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "champ_model_test_scores.html"

    summary_df = pd.DataFrame(
        [
            {
                "Champion_Model": champ_result["model_name"],
                "AMSE": champ_result["AMSE"],
                "Test_MSE": test_metrics["MSE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_MAE": test_metrics["MAE"],
                "Test_R2": test_metrics["R2"],
                "Test_Adjusted_R2": test_metrics["Adjusted_R2"],
            }
        ]
    )

    params_df = pd.DataFrame(
        [{"hyperparameter": k, "value": v} for k, v in champ_result["best_params"].items()]
    )

    html = f"""
    <html>
      <head>
        <title>Champion Model Test Scores</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          h1, h2 {{ margin-bottom: 8px; }}
          table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
          th {{ background-color: #f4f4f4; }}
        </style>
      </head>
      <body>
        <h1>Champion Model Test Evaluation</h1>
        <p><b>Selection Criterion:</b> Lowest AMSE (Average MSE from 5-fold stratified CV)</p>
        <h2>Champion Summary</h2>
        {summary_df.to_html(index=False)}
        <h2>Champion Hyperparameters</h2>
        {params_df.to_html(index=False)}
      </body>
    </html>
    """

    output_file.write_text(html, encoding="utf-8")
    log("Saved champion model test HTML report to:", output_file)
    return output_file


def run_champion_selection() -> dict:
    log("Training all candidate models for champion selection")

    candidates = [
        run_linear_regression(),
        run_elastic_net(),
        run_random_forest(),
        run_xgboost_regressor(),
    ]

    champ = min(candidates, key=lambda item: item["AMSE"])
    champ_model_path = _save_model_pickle(champ["estimator"], "champ.pkl")

    log("Champion model selected:", champ["model_name"])
    log("Champion AMSE:", champ["AMSE"])
    log("Champion hyperparameters:", champ["best_params"])

    X_test, y_test = load_test_data()
    y_pred_test = champ["estimator"].predict(X_test)
    test_metrics = compute_regression_metrics(y_test, y_pred_test, n_features=X_test.shape[1])

    for key in ["MSE", "RMSE", "MAE", "R2", "Adjusted_R2"]:
        log(f"Champion test {key}: {test_metrics[key]}")

    report_path = _save_champ_test_html(champ, test_metrics, CHAMP_TEST_DIR)

    return {
        "champion": champ,
        "test_metrics": test_metrics,
        "report_path": report_path,
        "champ_model_path": champ_model_path,
    }


if __name__ == "__main__":
    run_champion_selection()
