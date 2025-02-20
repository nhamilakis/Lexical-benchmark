#!/usr/bin/env python
import argparse
import json
import sys
import typing as t
from pathlib import Path

from lexical_benchmark import lb_types, train_lib
from lexical_benchmark.datasets import DataSchemaType, DatasetLoader, child_realistic, stella
from lexical_benchmark.utils import generic as generic_utils

DatasetType = t.Literal["childrealistic", "stela"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model training and reporting tool")
    # Create parent parser with shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=Path.cwd(),
        help="Output directory for path file",
    )
    parent_parser.add_argument("--lang", type=str, default="by_month", help="Type of dataset architecture to use.")
    parent_parser.add_argument("--schema", type=DataSchemaType, default=["childrealistic", "stela"])
    parent_parser.add_argument("--model-type", nargs="+", default=["lstm", "transformer"])
    parent_parser.add_argument("--datasets", nargs="+", default=["childrealistic", "stela"])

    # Define subparser for root_parser
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Training subcommand
    train_parser = subparsers.add_parser(
        "train-args",
        help="Generate model training array",
        parents=[parent_parser],
    )
    train_parser.add_argument(
        "--manual-flags", type=str, help="Path to a csv file with extra model choice (skip/override/resume, ...)"
    )
    train_parser.add_argument("--output-type", choices=["json", "toml"], default="toml")
    # Report subcommand - add your report specific arguments here
    report_parser = subparsers.add_parser(
        "report",
        help="Generate reports from model results",
        parents=[parent_parser],
    )
    report_parser.add_argument("--output-type", choices=["csv", "json"], default="csv")
    return parser.parse_args()


def extract_model_args(
    dataset_name: DatasetType, lang: str, model_type: lb_types.ModelType, schema: DataSchemaType
) -> list[train_lib.TrainArgs]:
    """Crawl Dataset folders and figure out if models have been trained."""
    if dataset_name == "childrealistic":
        dataset: DatasetLoader = child_realistic.ChildRealisticDataset()
    elif dataset_name == "stela":
        dataset: DatasetLoader = stella.STELATranscriptDataset()

    to_train = []
    for item in dataset.iter_models(lang=lang, schema_type=schema, model_type=model_type):
        if item.needs_training:
            train_args = train_lib.TrainArgs.from_model(item, dataset_name=dataset_name)
            to_train.append(train_args)
    return to_train


def build_train_args(
    models: list[lb_types.ModelType], datasets: list[DatasetType], *, lang: str, schema: DataSchemaType
) -> dict[str, dict]:
    """Build list of args for training the models."""
    training_items: list[train_lib.TrainArgs] = []
    for md in models:
        for dt in datasets:
            training_items.extend(extract_model_args(dataset_name=dt, model_type=md, lang=lang, schema=schema))

    # TODO apply external filters

    # return as dict
    return {f"{idx}": ta.to_dict() for idx, ta in enumerate(training_items)}


def main() -> None:
    """Main function to generate training paths."""
    args = parse_args()
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    match args.command:
        case "train-args":
            train_args = build_train_args(
                models=args.model_type,
                datasets=args.datasets,
                lang=args.lang,
                schema=args.schema,
                external_flags=args.manual_flags,
            )
            file = output_dir / "train-args.index.xxx"
            if len(train_args) == 0:
                print("No models found for training.")
                sys.exit(0)

            if args.output_type == "json":
                with (file.with_suffix(".json")).open("w") as fp:
                    json.dump(train_args, fp)
            elif args.output_type == "toml":
                generic_utils.write_toml(train_args, file.with_suffix(".toml"))

            print(f"Saved index to {file}.")

        case "report":
            ...


if __name__ == "__main__":
    main()
