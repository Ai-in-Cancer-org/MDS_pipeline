import traceback
import time
import torch
from utils_mlflow import MlflowTracker
from train import inner_main
from parse import parse_args

def run():
    args = parse_args()
    run_uuid = args.run_uuid
    metric = 0
    user_attr = {}
    system_attr = {}

    for i in range(args.cv_start, args.nr_cv):
        print(f"START CV {i}")
        try:
            metric += inner_main(args, i, trial=None, resume=bool(run_uuid))
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error in CV {i}: {e}")
        print(f"END CV {i}")

    print("Final Avg Metric:", metric / (args.nr_cv - args.cv_start))

if __name__ == "__main__":
    run()
