"""
convert word count into accumulated monthly count
"""
import argparse
import os
import sys
from pathlib import Path
from lm_benchmark.analysis.score import MonthCounter

ROOT = '/Users/jliu/PycharmProjects/Lexical-benchmark/data/eval/exp'

def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_file",
                        default=f'{ROOT}/score/generation.csv')
    parser.add_argument("--est_file",
                        default=f'{ROOT}/score/vocal_month.csv')
    parser.add_argument("--lang",default='AE')
    parser.add_argument("--set", default='machine')
    parser.add_argument("--header",default='unprompted_0.3')
    parser.add_argument("--threshold_lst",default=[60])
    return parser.parse_args()


def main():
    """ Run the GoldReference loader and write results to a file """
    args = arguments()
    gen_file = Path(args.gen_file)
    est_file = Path(args.est_file)
    lang = args.lang
    set = args.set
    test_file = Path(f'{ROOT}/test/test_set/'+lang+'_'+set+'1.csv')
    count_test_file = Path(f'{ROOT}/score/'+ lang+f'/count_test/{args.header}.csv')
    count_all_file = Path(f'{ROOT}/score/count_all/{args.header}.csv')
    score_file = f'{ROOT}/score/'+lang+'/score/'
    if not os.path.exists(score_file):
        os.makedirs(score_file)

    for threshold in args.threshold_lst:
        # get freq grouped by month
        count_loader = MonthCounter(
            gen_file=gen_file,
            est_file=est_file,
            test_file=test_file,
            count_all_file=count_all_file,
            count_test_file = count_test_file,
            header=args.header,
            threshold=threshold
        )
        count_score = count_loader.get_score()
        count_score.to_csv(score_file+args.header+'_'+str(threshold)+'.csv')




if __name__ == "__main__":
    args = sys.argv[1:]
    main()
