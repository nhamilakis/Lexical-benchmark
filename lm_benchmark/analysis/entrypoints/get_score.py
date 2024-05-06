"""
convert word count into accumulated monthly count
"""
import argparse
import sys
from pathlib import Path
from lm_benchmark.analysis.score import MonthCounter


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/generation.csv')
    parser.add_argument("--est_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/vocal_month.csv')
    parser.add_argument("--test_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/test_set/AE_human.csv')
    parser.add_argument("--count_all_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/count_all/content.csv')
    parser.add_argument("--count_test_file",
                        default='/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/count_test/AE_human.csv')
    parser.add_argument("--header",default='content')
    parser.add_argument("--threshold", default=6)
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    gen_file = Path(args.gen_file)
    est_file = Path(args.est_file)
    test_file = Path(args.test_file)
    count_all_file = Path(args.count_all_file)
    count_test_file = Path(args.count_test_file)
    header = args.header

    # get freq grouped by month
    count_loader = MonthCounter(
        gen_file=gen_file,
        est_file=est_file,
        test_file=test_file,
        count_all_file=count_all_file,
        header=header,
        threshold=args.threshold
    )

    adjusted_df = count_loader.get_count()
    adjusted_df.to_csv(count_test_file, index=False)
    # select from the eval test



if __name__ == "__main__":
    args = sys.argv[1:]
    main()

'''
import pandas as pd
merged = pd.read_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/score/count_all/content.csv')
test = pd.read_csv('/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp/test/test_set/AE_human.csv')

merged_df = pd.merge(merged, test, how='inner', left_index=True, right_on='word')
merged_df = pd.merge(merged_df, df2, on='word', how='outer')



        # Select rows where 'Index_in_df1' column is not null
        self._selected_rows = merged_df[merged_df['word'].notnull()]

'''

import pandas as pd

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}

df = pd.DataFrame(data)

# Calculate cumulative count by rows starting from the second column to the last
cumulative_count = df.iloc[:, 1:].cumsum(axis=1)

print(cumulative_count)

