import os
import logging
import xgboost
import sklearn.metrics
import pandas as pd
import argparse

from typing import Any, Dict, List, Tuple
from xgboost.callback import _Model, TrainingCallback


class TrainingLogger(TrainingCallback):
    def __init__(self, iters_per_log: int, iters_count: int) -> None:
        super().__init__()
        self._iters_per_log = iters_per_log
        self._iters_count = iters_count

    def after_iteration(self, model: Any, epoch: int, evals_log: Dict[str, Dict[str, List[float] | List[Tuple[float, float]]]]) -> bool:
        if epoch % self._iters_per_log == 0:
            logging.info(f"iter: {epoch} / {self._iters_count}")
            logging.info(f"accuracy={evals_log['eval']['auc'][-1]}")
        return False


def train(model: Any, params, x_train, x_evals, y_train, y_evals):
    xgboost.train(
        params=params,
        xgb_model=model, 
        evals=x_evals, 
        evals_result=y_evals, 
        d_train=x_train
    )

def test(model: Any, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_pred, y_test)

    logging.info(f"accuracy={accuracy}")

def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG
    )

    parser = argparse.ArgumentParser(description="XGBoost")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--ne", type=int, default=2000, metavar="NE", 
                        help="n estimators (default:1000)")
    parser.add_argument("--rs", type=int, default=1, metavar="RS",
                        help="random state (default: 1)")
    parser.add_argument("--booster", type=str, choices=["gbtree", "gblinear", "dart"], default="gbtree", 
                        help="Choose the booster", metavar="B")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "cuda"], default="cuda", 
                        help="Choose the device", metavar="DEV")
    parser.add_argument("--x_train_path", type=str, default="datasets_train",
                        help="Assign the path of x_train_csv", metavar="x-train-path")
    parser.add_argument("--x_evals_path", type=str, default="datasets_eval",
                        help="Assign the path of x_train_csv", metavar="x-eval-path")
    parser.add_argument("--x_test_path", type=str, default="datasets_test",
                        help="Assign the path of x_train_csv", metavar="x-test-path")
    parser.add_argument("--y_train_path", type=str, default="datasets/y_train.csv",
                        help="Assign the path of x_train_csv", metavar="y-train-path")
    parser.add_argument("--y_evals_path", type=str, default="datasets_eval",
                        help="Assign the path of x_train_csv", metavar="y-eval-path")
    parser.add_argument("--y_test_path", type=str, default="datasets/y_test.csv",
                        help="Assign the path of x_train_csv", metavar="y-test-path")
    parser.add_argument("--save_model", type=bool, default=False, 
                        help="Save model or not", metavar="save-model")
    parser.add_argument("--model_folder_path", type=str, default="models", 
                        help="The folder to save the model", metavar="model-folder-path")

    args = parser.parse_args()

    file_paths = os.listdir("datasets")