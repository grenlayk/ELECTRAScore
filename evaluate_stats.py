import argparse

from analysis import SUMStat


def main(dataset):
    if dataset == "Newsroom" or dataset == "SummEval":
        summ_stat = SUMStat(f"data/{dataset}/scores.pkl")
        summ_stat.evaluate_summary("fluency")
    else:
        print("You used wrong dataset. Please choose Newsroom or SummEval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scorer parameters")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to calculate statistics. Newsroom or SummEval",
    )

    args = parser.parse_args()
    main(args.dataset)
