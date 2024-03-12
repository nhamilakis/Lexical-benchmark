import argparse
import sys
from pathlib import Path
from typing import Literal

import dataclasses


LANG_TYPE = Literal['AE', 'BE', 'FR']


@dataclasses.dataclass
class Arguments:
    text_path: Path  # Root path to the Childes transcripts
    output_path: Path = Path('data/output')  # Location for output
    eval_path: Path = Path('data/eval')

    input_condition: Literal['recep', 'exp'] = 'recep'  # Types of vocabulary
    language: LANG_TYPE = 'BE'  # Language to use
    eval_type: Literal["CDI", "Wuggy_filtered"] = "CDI"  # Type to test
    eval_condition: Literal["recep", "exp"] = "exp"  # Condition ? words to evaluate

    hour: int = 3  # Estimation of hours per day
    words_per_hour: int = 10000  # Estimated number of words per hour
    threshold_range: tuple[int, ...] = (1, 200, 600, 1000)  # threshold to decide knowing a productive word or not

    @classmethod
    def from_cmd(cls, argv: list[str] = None):
        argv = argv if argv else sys.argv[1:]
        parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')

        parser.add_argument('--language', type=str, default='BE',
                            help='langauges to test: AE, BE or FR')

        parser.add_argument('--eval-type', type=str, default='CDI',
                            help='langauges to test: CDI or Wuggy_filtered')

        parser.add_argument('--eval-condition', type=str, default='exp',
                            help='which type of words to evaluate; recep or exp')

        parser.add_argument('--text-path', type=str, default='CHILDES',
                            help='root Path to the CHILDES transcripts; '
                                 'one of the variables to invetigate')

        parser.add_argument('--output-path', type=str, default='Output',
                            help='Path to the freq output.')

        parser.add_argument('--input-condition', type=str, default='recep',
                            help='types of vocab: recep or exp')

        parser.add_argument('--hour', type=int, default=3,
                            help='the estimated number of hours per day')

        parser.add_argument('--words-per-hour', type=int, default=10000,
                            help='the estimated number of words per hour')

        parser.add_argument('--threshold-range', type=tuple, default=(1, 200, 600, 1000),
                            help='threshold to decide knowing a productive word or not, '
                                 'one of the variable to invetigate')

        parser.add_argument('--eval_path', type=str, default='Human_eval/',
                            help='path to the evaluation material; one of the variables to investigate')

        args = parser.parse_args(argv)
        return cls(**vars(args))
