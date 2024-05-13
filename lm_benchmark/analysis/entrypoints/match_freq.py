import argparse
import sys
from pathlib import Path
from lm_benchmark.analysis.freq import FreqMatcher


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_src",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/corpus/BE.csv')
    parser.add_argument("--CHILDES_src",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/freq/CHILDES.csv')
    parser.add_argument("--machine_src",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/corpus/freq/3200.csv')
    parser.add_argument("--human_target",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/corpus/BE_human.csv')
    parser.add_argument("--machine_target",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/corpus/BE_machine.csv')
    parser.add_argument("--header",default='word')
    parser.add_argument("--freq_header", default='freq_m')
    parser.add_argument("--num_bins", default=6)
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    machine_src = Path(args.machine_src)
    human_src = Path(args.human_src)
    CHILDES_src = Path(args.CHILDES_src)
    machine_target = Path(args.machine_target)
    human_target = Path(args.human_target)
    # match files
    freq_matcher = FreqMatcher(
        human=human_src,
        CHILDES=CHILDES_src,
        machine=machine_src,
        num_bins=args.num_bins,
        header=args.header,
        freq_header=args.header
    )

    matched_CDI, matched_audiobook,human_stat,machine_stat = freq_matcher.get_matched_data()
    matched_CDI.to_csv(human_target)
    matched_audiobook.to_csv(machine_target)

    human_stat.to_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/freq_stat/BE_human.csv')
    machine_stat.to_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/freq_stat/BE_machine.csv')


if __name__ == "__main__":
    args = sys.argv[1:]
    main()


