import pandas as pd
import datasets
from train_eval_script import filter_train_and_evaluate


def correcting_func(texts):
    # returns list of corrected scores for list of texts
    df = pd.read_csv('data/tmp/labeled_generated_data.csv')
    return df['chatgpt_label'].to_list()


augmentations = datasets.load_dataset("csv", data_files="data/tmp/generated_data.csv")
filter_train_and_evaluate(
    correcting_func,
    augmentations['train'],
    num_pos_to_leave="equal",
    device='mps:0',
    epochs=1,
)
