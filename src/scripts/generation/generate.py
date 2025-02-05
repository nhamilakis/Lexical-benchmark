#!/usr/bin/env python
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lexical_benchmark.settings import chunk2month
from lexical_benchmark.utils import slurm_utils
from lexical_benchmark.utils.gen_util import BatchProcessor, Logger, TextGenerator

slurm_utils.info_header()


def parse_args():
    # Run parameters
    parser = argparse.ArgumentParser(description="Finetune decoder-onlyls models")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/models/STELATranscriptions2/by_month/EN/10/00/trans",
        help="Path to the base LM",
    )
    parser.add_argument(
        "--generation_path",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/gen/merged2/STELATranscriptions2/by_month/EN/10/00/trans",
        help="Path to the generated texts",
    )
    parser.add_argument(
        "--gen_file",
        type=str,
        default="/scratch1/projects/lexical-benchmark/v2/gen/merged/CHILDES_model.csv",
        help="Path to the generated texts",
    )
    parser.add_argument("--temp_lst", type=list, default=[0.3, 0.6, 1.0, 1.5], help="target month model")
    parser.add_argument("--hour_per_year", default=1000, type=int, help="Estimated yearly exposure hours")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--added_tokens", default=["'", "|"], help="A list of added special tokens")
    parser.add_argument("--use_vllm",  action="store_true", help="if true, apply vllm for transformer model")
    parser.add_argument("--save_interval", default=1024, type=int, help="The number of rows to save")
    parser.add_argument("--resume", action="store_true", help="if true, resume from intermediate generation")
    parser.add_argument("--override", action="store_true", help="if true, erase previous generation and replace it.")
    parser.add_argument("--debug", action="store_true", help="if debug, generate first 10 sentences")
    return parser.parse_args()


def main(args):
    """Main function to run the generation process with the specified arguments."""
    try:
        # Setup paths
        generation_path = Path(args.generation_path)
        generation_path.mkdir(parents=True, exist_ok=True)

        model_type = Path(args.generation_path).name
        # automaitically enable vllm if there is transformer model
        use_vllm = "trans" in args.model_path.lower() if args.use_vllm else False
        # Get month from path
        try:
            # convert the chunk_num to month
            chunk_num = int(Path(args.model_path).parents[1].name)
            month = chunk2month(chunk_num,args.hour_per_year)
        except ValueError as e:
            raise ValueError(f"Parent folder of {args.generation_path} does not contain month info!") from e
        print(f"{month=}")

        # Setup logger
        logger = Logger.setup_stdout(debug=args.debug)
        logger.info(f"Starting generation with arguments: {args}")
        logger.info(f"Detected model type: {model_type}, vLLM enabled: {use_vllm}")

        # Determine if we're using multiple GPUs
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load and filter data by month
        df = pd.read_csv(args.gen_file).loc[:, "month":]
        df = df[df["model"] == month]
        logger.info(f"Loaded input file with {len(df)} rows for month {month}")

        # Debug mode handling
        if args.debug:
            df = df.head(20)
            logger.info("Debug mode: using first 20 rows only")

        # Create generator
        generator = TextGenerator(
            model_path=args.model_path,
            model_type=model_type,
            use_vllm=use_vllm
        )

        # Add special tokens
        if len(args.added_tokens) > 0:
            generator.add_special_tokens(args.added_tokens)

        # Create processor
        processor = BatchProcessor(
            generator=generator,
            save_path=generation_path,
            chunk_size=args.save_interval,
            hour_per_year=args.hour_per_year,
            debug=args.debug,
        )

        # Check resume
        if not args.resume and processor.get_save_file(intermidiate=True, debug=args.debug).is_file() and not args.override:
            print(
                "ERROR: current folder has intermiate file but no override or resume flag was passed\n",
                file=sys.stderr
            )
            raise ValueError(f"Failed: {processor.get_save_file()}")

        # Check target
        if processor.get_save_file(debug=args.debug).is_file() and not args.override:
            print(
                    "ERROR: current folder target file already exists and no override flag was passed\n",
                    file=sys.stderr
                )
            raise ValueError(f"Failed: {processor.get_save_file()}")


        # Process data using BatchProcessor
        result_df = processor.process_dataframe(
            prompt_df=df,
            temp_lst=args.temp_lst,
            resume=args.resume
        )
        # Save results
        if result_df is not None:
            final_path = processor.get_save_file(debug=args.debug)
            result_df.to_csv(final_path)
            logger.info(f"Generation completed. Final results saved to {final_path}")
    finally:
        torch.cuda.empty_cache()




if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    finally:
        slurm_utils.info_footer()


