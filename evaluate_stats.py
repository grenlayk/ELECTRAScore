import argparse

from analysis import SUMStat


def main(dataset, input_file, dataset_level):
    if dataset in ['Newsroom', 'SummEval', 'JFLEG']:
        summ_stat = SUMStat(f'data/{dataset}/{input_file}')
        print("Evaluation of ", dataset, " dataset:")
        if dataset == 'JFLEG':
            summ_stat.evaluate('fluency', dataset_level=dataset_level)
        else:
            summ_stat.evaluate_summary('fluency', dataset_level=False)
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
    parser.add_argument(
        '--dataset_level', type=bool, required=False,
        default=False,
        help='Type of aggregation for JFLEG dataset.'
    )

    args = parser.parse_args()
    main(args.dataset, args.input_file, args.dataset_level)
