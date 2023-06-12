import argparse

from analysis import SUMStat


def main(dataset, input_file):
    if dataset in ['Newsroom', 'SummEval', 'JFLEG']:
        summ_stat = SUMStat(f'data/{dataset}/{input_file}')
        print("Evaluation of ", dataset, " dataset:")
        dataset_level = True if dataset == 'JFLEG' else False
        summ_stat.evaluate_summary('fluency', dataset_level=dataset_level)
    else:
        print('You used wrong dataset. \
              Please choose Newsroom, SummEval or JFLEG.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument(
        '--dataset', type=str, required=True,
        help='The dataset to calculate statistics. Newsroom, SummEval or JFLEG'
    )
    parser.add_argument(
        '--input_file', type=str, required=False,
        default='scores.pkl',
        help='Name of file to evaluate in dataset folder. Default: scores.pkl'
    )

    args = parser.parse_args()
    main(args.dataset, args.input_file)
