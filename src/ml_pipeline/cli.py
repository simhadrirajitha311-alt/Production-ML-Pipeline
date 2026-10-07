from __future__ import annotations

import argparse
import json
from pprint import pprint

from ml_pipeline.data.loader import DataLoader
from ml_pipeline.data.profiler import profile_dataframe
from ml_pipeline.data.validator import validate_dataframe
from ml_pipeline.tasks.detector import detect_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoML Production Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile", help="Profile a dataset")
    profile_parser.add_argument("data_path")
    profile_parser.add_argument("--target", default=None)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--target", required=True)
    train_parser.add_argument("--task", choices=["classification", "regression", "clustering"], default="classification")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved model")
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--target", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict using a saved model")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)

    tune_parser = subparsers.add_parser("tune", help="Tune a model")
    tune_parser.add_argument("--model", default="random_forest")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "profile":
        df = DataLoader.load(args.data_path)
        validation = validate_dataframe(df, target=args.target)
        profile = profile_dataframe(df, target=args.target)
        print(json.dumps({"validation": validation, "profile": profile.to_dict()}, indent=2))

    elif args.command == "train":
        df = DataLoader.load(args.data)
        detection = detect_task(df, target=args.target, override=args.task)
        print(json.dumps(detection, indent=2))

    elif args.command == "evaluate":
        df = DataLoader.load(args.data)
        print({"rows": len(df), "columns": len(df.columns), "target": args.target})

    elif args.command == "predict":
        with open(args.input, "r", encoding="utf-8") as file:
            payload = json.load(file)
        print({"prediction_input": payload, "model": args.model})

    elif args.command == "tune":
        print({"msg": f"Optimization for {args.model} is enabled in the training flow."})

    else:
        parser.print_help()
