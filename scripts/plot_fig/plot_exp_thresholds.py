import os
from util import load_transcript, get_freq, count_by_month, get_score
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Investigate CHILDES corpus')

    parser.add_argument('--lang', type=str, default='BE',
                        help='langauges to test: AE, BE or FR')

    parser.add_argument('--eval_type', type=str, default='CDI',
                        help='langauges to test: CDI or Wuggy_filtered')

    parser.add_argument('--eval_condition', type=str, default='exp',
                        help='which type of words to evaluate; recep or exp')

    parser.add_argument('--TextPath', type=str, default='CHILDES',
                        help='root Path to the CHILDES transcripts; one of the variables to invetigate')

    parser.add_argument('--OutputPath', type=str, default='Output',
                        help='Path to the freq output.')

    parser.add_argument('--input_condition', type=str, default='exp',
                        help='types of vocab: recep or exp')

    parser.add_argument('--hour', type=dict, default=10,
                        help='the estimated number of waking hours per day; data from Alex(2023)')

    parser.add_argument('--word_per_sec', type=int, default=3,
                        help='the estimated number of words per second')

    parser.add_argument('--sec_frame_path', type=str, default='vocal_month.csv',
                        help='the estmated vocalization seconds per hour by month')

    parser.add_argument('--threshold_range', type=list, default=[50, 100, 200, 300],
                        help='threshold to decide knowing a productive word or not, one of the variable to invetigate')

    parser.add_argument('--eval_path', type=str, default='Human_eval/',
                        help='path to the evaluation material; one of the variables to invetigate')

    return parser.parse_args(argv)



def plot_thresholds(OutputPath, eval_path, threshold_range, eval_condition, freq_frame, hour, lang,
                  eval_type):
    sns.set_style('whitegrid')

    eval_dir = eval_path + 'CDI' + '/' + lang + '/' + eval_condition

    # plot the CDI results
    # load multiple files
    for file in os.listdir(eval_dir):
        selected_words = pd.read_csv(eval_dir + '/' + file).iloc[:, 5:-6]

    size_lst = []
    month_lst = []

    n = 0
    while n < selected_words.shape[0]:
        size_lst.append(selected_words.iloc[n])
        headers_list = selected_words.columns.tolist()
        month_lst.append(headers_list)
        n += 1

    size_lst_final = [item for sublist in size_lst for item in sublist]
    month_lst_final = [item for sublist in month_lst for item in sublist]
    month_lst_transformed = []
    for month in month_lst_final:
        month_lst_transformed.append(int(month))
    # convert into dataframe
    data_frame = pd.DataFrame([month_lst_transformed, size_lst_final]).T
    data_frame.rename(columns={0: 'month', 1: 'Proportion of acquired words'}, inplace=True)
    data_frame_final = data_frame.dropna(axis=0)

    ax = sns.lineplot(x="month", y="Proportion of acquired words", data=data_frame_final, color='black', linewidth=2.5,
                      label=lang + '_CDI')

    # set the limits of the x-axis for each line
    for line in ax.lines:
        plt.xlim(0, 36)
        plt.ylim(0, 1)
    mean_value_CDI = data_frame_final.groupby('month')['Proportion of acquired words'].mean()

    # loop over thresholds
    for threshold in threshold_range:

        avg_values_lst = []
        # averaged by different groups
        for freq in set(list(freq_frame['group_original'].tolist())):
            word_group = freq_frame[freq_frame['group_original'] == freq]
            score_frame, avg_value = get_score(word_group, OutputPath, threshold, hour)
            avg_values_lst.append(avg_value.values)

        arrays_matrix = np.array(avg_values_lst)

        # Calculate the average array along axis 0
        avg_values = np.mean(arrays_matrix, axis=0)

        # Plotting the line curve
        ax = sns.lineplot(score_frame.columns, avg_values, label='threshold: ' + str(threshold))

    plt.title('{} CHILDES {} vocab(tested on {})'.format(lang, eval_condition, eval_type), fontsize=15)
    # plt.title('Accumulator on {} CHILDES ({} vocab)'.format(lang,eval_condition), fontsize=15)
    plt.xlabel('age in month', fontsize=15)
    plt.ylabel('Proportion of children', fontsize=15)

    plt.tick_params(axis='both', labelsize=10)

    plt.legend()

    figure_path = OutputPath + '/Figures/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.savefig(figure_path + '/Curve.png', dpi=800)
    plt.show()


def main(argv):
    # Args parser
    args = parseArgs(argv)

    TextPath = args.TextPath
    eval_condition = args.eval_condition
    input_condition = args.input_condition
    lang = args.lang
    eval_type = args.eval_type
    OutputPath = args.OutputPath + '/' + eval_type + '/' + lang + '/' + eval_condition
    eval_path = args.eval_path
    hour = args.hour
    threshold_range = args.threshold_range
    word_per_sec = args.word_per_sec
    sec_frame = pd.read_csv(args.sec_frame_path)

    # step 1: load data and count words

    freq_frame = pd.read_csv(OutputPath, month_stat, eval_path, hour, word_per_sec, eval_type, lang, eval_condition,
                             sec_frame)

    # step 2: get the score based on different thresholds
    plot_thresholds(OutputPath, eval_path, threshold_range, eval_condition, freq_frame, hour, lang, eval_type)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)