from typing import Any, Dict, List, Literal, Tuple
from xgboost.callback import TrainingCallback
from datetime import datetime, timezone, timedelta

import argparse
import logging
import xgboost
import joblib
import pandas as pd


class CustomCallback(TrainingCallback):
    def __init__(self) -> None:
        super().__init__()

    def after_iteration(self, model: Any, epoch: int, evals_log: Dict[str, Dict[str, List[float] | List[Tuple[float, float]]]]) -> bool:
        accuracy = 1 - evals_log['test']['error'][-1]
        logging.info(f"epoch={epoch}")
        logging.info(f"accuracy={accuracy}")

    def after_training(self, model: Any) -> Any:
        best_accuracy = 1 - model.best_score
        best_epoch = model.best_iteration

        logging.info("Training ends!")
        logging.info(f"best_epoch={best_epoch}")
        logging.info(f"best_accuracy={best_accuracy}")

        return model
    

def train(
    xgtrain: xgboost.DMatrix, 
    xgtest: xgboost.DMatrix, 
    learning_rate: float = 0.3, 
    n_estimators: int = 2000, 
    random_state: int = 42, 
    early_stopping_rounds: int = 1000, 
    booster: Literal['gbtree', 'gblinear', 'dart'] = 'gbtree', 
    device: Literal['cpu', 'gpu', 'cuda'] = 'cuda'
):
    params = {
        'booster': booster, 
        'device': device, 
        'eta': learning_rate, 
        'objective': 'binary:logistic', 
        'eval_metric': 'error', 
        'seed': random_state
    }

    watchlist  = [(xgtest,'test')]

    evals_result = {}

    model = xgboost.train(
        params=params, 
        dtrain=xgtrain, 
        num_boost_round=n_estimators, 
        evals=watchlist, 
        evals_result=evals_result, 
        early_stopping_rounds=early_stopping_rounds, 
        verbose_eval=0, 
        callbacks=[CustomCallback()]
    )

    return model

def main():
    '''
    Default: read datasets from /mnt/datasets.
    These files should exist.
    + /mnt/datasets/x_train.csv
    + /mnt/datasets/y_train.csv
    + /mnt/datasets/x_test.csv
    + /mnt/datasets/y_test.csv
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="XGBoost")
    parser.add_argument("--lr", type=float, default=0.3, metavar="LR",
                        help="learning rate (default: 0.3)")
    parser.add_argument("--ne", type=int, default=2000, metavar="NE", 
                        help="n estimators (default:2000)")
    parser.add_argument("--rs", type=int, default=42, metavar="RS",
                        help="random state (default: 42)")
    parser.add_argument("--esp", type=int, default=1000, metavar="ESP", 
                        help="early stopping rounds (default: 1000)")
    parser.add_argument("--booster", type=str, choices=["gbtree", "gblinear", "dart"], default="gbtree", 
                        help="Choose the booster", metavar="B")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "cuda"], default="cuda", 
                        help="Choose the device", metavar="DEV")
    parser.add_argument("--x_train_path", type=str, default="datasets/x_train.csv",
                        help="Assign the path of x_train_csv", metavar="x-train-path")
    parser.add_argument("--x_test_path", type=str, default="datasets/x_test.csv",
                        help="Assign the path of x_train_csv", metavar="x-test-path")
    parser.add_argument("--y_train_path", type=str, default="datasets/y_train.csv",
                        help="Assign the path of x_train_csv", metavar="y-train-path")
    parser.add_argument("--y_test_path", type=str, default="datasets/y_test.csv",
                        help="Assign the path of x_train_csv", metavar="y-test-path")
    parser.add_argument("--save_model", type=bool, default=False, 
                        help="Save model or not", metavar="save-model")
    parser.add_argument("--model_folder_path", type=str, default="models", 
                        help="The folder to save the model", metavar="model-folder-path")

    args = parser.parse_args()

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG
    )

    x_train_df = pd.read_csv(args.x_train_path, header=0)
    y_train_df = pd.read_csv(args.y_train_path, header=0)
    x_test_df = pd.read_csv(args.x_test_path, header=0)
    y_test_df = pd.read_csv(args.y_test_path, header=0)
    
    xgtrain = xgboost.DMatrix(x_train_df.values, y_train_df.values)
    xgtest = xgboost.DMatrix(x_test_df.values, y_test_df.values)

    logging.info(f"Trying to use device: {args.device}")
    logging.info(f"learning_rate={args.lr}")
    logging.info(f"n_estimators={args.ne}")
    logging.info(f"random_state={args.rs}")
    logging.info(f"booster={args.booster}")

    model = train(
        xgtrain=xgtrain, 
        xgtest=xgtest, 
        learning_rate=args.lr, 
        n_estimators=args.ne, 
        random_state=args.rs, 
        booster=args.booster, 
        device=args.device
    )

    if args.save_model is True and model is not None:
        lr_str = "lr-" + str(args.lr)
        ne_str = "ne-" + str(args.ne)
        rs_str = "rs-" + str(args.rs)
        booster_str = "booster-" + str(args.booster)
        current_time_str = datetime.now(tz=timezone(offset=timedelta(hours=8))).strftime("%Y-%m-%d-%H-%M-%S")
        str_list = [lr_str, ne_str, rs_str, booster_str, current_time_str]
        model_name = "_".join(str_list) + ".pkl"
        model_folder_path_processed = str(args.model_folder_path)
        while model_folder_path_processed.endswith("/"):
            model_folder_path_processed = model_folder_path_processed.removesuffix("/")
        model_path = f"{model_folder_path_processed}/{model_name}"

        joblib.dump(model, model_path)

        logging.info(f"model_path={model_path}")

if __name__ == '__main__':
    main()