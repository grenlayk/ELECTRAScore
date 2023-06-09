import pandas as pd
import datasets
from train_eval_script import filter_train_and_evaluate, evaluate_all, score_dataset_with_model


def correcting_func(texts):
    # returns list of corrected scores for list of texts
    df = pd.read_csv('data/tmp/labeled_generated_data.csv')
    return df['chatgpt_label'].to_list()


device = 'mps:0'
newsroom_output = 'newsroom_scores.pkl'
summeval_output = 'summeval_scores.pkl'
augmentations = datasets.load_dataset("csv", data_files="data/tmp/generated_data.csv")

# filter_train_and_evaluate(correcting_func, augmentations['train'], num_pos_to_leave="equal", device='mps:0', epochs=1)

# model = 
# tokenizer = 
# score_dataset_with_model(tokenizer, models=[model],
#         metrics_names=['electra_score_akim'],
#         name='Newsroom', device=device, output=newsroom_output)
# score_dataset_with_model(tokenizer, models=[model],
#         metrics_names=['electra_score_akim'],
#         name='SummEval', device=device, output=newsroom_output)
evaluate_all(summeval_output=summeval_output, newsroom_output=newsroom_output)
