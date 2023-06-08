from typing import List, Union
from datasets import Dataset, load_dataset, load_metric, concatenate_datasets
import pandas as pd
from collections import Counter
import transformers as ts

from score import Scorer
from analysis import SUMStat


def prepare_setup(data, tokenizer):
    cola = load_dataset('glue', 'cola', cache_dir='tmp/data')

    # tokenize train data
    def tokenizing_fn(instance):
        return tokenizer(instance['text'], truncation=True)
    tokenized_train = data.map(tokenizing_fn, batched=True)
    tokenized_train = tokenized_train.remove_columns(['text', 'id'])

    # tokenize validation data
    def tokenizing_fn_val(instance):
        return tokenizer(instance['sentence'], truncation=True)
    tokenized_val = cola['validation'].map(tokenizing_fn_val, batched=True)
    tokenized_val = tokenized_val.remove_columns(['sentence', 'idx'])

    return tokenized_train, tokenized_val


def merge_data(cola, data_augm, num_pos_to_leave):
    if type(num_pos_to_leave) == str:
        if num_pos_to_leave == 'equal':
            num_pos_to_leave_int = (
                Counter(cola['label'])[0]
                + Counter(data_augm['label'])[0]
                - Counter(cola['label'])[1]
            )
        else:
            raise NotImplementedError
    else:
        num_pos_to_leave_int = num_pos_to_leave

    data_augm = concatenate_datasets([
        data_augm.filter(lambda row: row['label'] == 0),
        data_augm.filter(lambda row: row['label'] == 1).select(
            range(num_pos_to_leave_int))
    ])

    return concatenate_datasets([data_augm, cola])


def filter_train_and_evaluate(
    correcting_func,
    augmentations: Dataset,
    num_pos_to_leave: Union[str, int] = "equal",
    learning_rate_e: int = 2e-5,
    learning_rate_ecl: int = 1e-5,
    train_batch_size: int = 32,
    val_batch_size: int = 150,
    epochs: int = 5,
    device: str = 'cuda:0',
    summeval_output: str = 'summeval_scores.pkl',
    newsroom_output: str = 'newsroom_scores.pkl',
):
    texts = augmentations["text"]
    # Predictions of LLM
    print('Starting scoring with provided model.')
    corrected_labels = correcting_func(texts)
    print('Finished scoring with provided model.')

    ### Here two options of saving augmentations
    print('Starting CoLA-E and CoLA-ECL creation.')
    cola_e_augm = augmentations.add_column("model_label", corrected_labels)
    cola_e_augm = cola_e_augm.filter(
        lambda row: row["label"] == row["model_label"])
    cola_e_augm = cola_e_augm.remove_columns("model_label")

    cola_ecl_augm = augmentations.add_column("model_label", corrected_labels)
    cola_ecl_augm = cola_ecl_augm.remove_columns("label")
    cola_ecl_augm = cola_ecl_augm.rename_column("model_label", "label")

    ### Here concatenation with original CoLA
    cola = pd.read_csv('data/tmp/in_domain_train.tsv', sep='\t', header=None)
    cola = cola.drop(columns=[2]).rename(columns={0: 'id', 1: 'label', 3: 'text'})
    cola = cola[['text', 'label', 'id']]
    cola = Dataset.from_pandas(cola)

    cola_e = merge_data(cola, cola_e_augm, num_pos_to_leave)
    cola_e.to_csv(f'data/tmp/cola_e.csv', index=False)

    cola_ecl = merge_data(cola, cola_ecl_augm, num_pos_to_leave)
    cola_ecl.to_csv(f'data/tmp/cola_ecl.csv', index=False)

    print('Finished CoLA-E and CoLA-ECL creation. \
          CoLA-E saved to data/tmp/cola_e.csv. \
          CoLA-ECL saved to data/tmp/cola_ecl.csv.')
    print('Class balance for CoLA-E: ', Counter(cola_e['label']))
    print('Class balance in CoLA-ECL: ', Counter(cola_ecl['label']))

    ### Here ELECTRA training
    print('Starting setup for training.')
    tokenizer = ts.AutoTokenizer.from_pretrained(
        'google/electra-large-discriminator', cache_dir='tmp/tokenizer')
    data_collator = ts.DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy_metric = load_metric('accuracy', cache_dir='tmp/metrics')
    cola_metric = load_metric('glue', 'cola', cache_dir='tmp/metrics')

    def compute_metrics(outputs):
        logits, labels = outputs
        preds = logits.argmax(1)

        metrics_dict = accuracy_metric.compute(
            references=labels, predictions=preds)
        metrics_dict.update(cola_metric.compute(
            references=labels, predictions=preds))

        return metrics_dict

    callbacks = [ts.EarlyStoppingCallback(3)]

    tokenized_train_e, tokenized_val_e = prepare_setup(cola_e, tokenizer)
    tokenized_train_ecl, tokenized_val_ecl = prepare_setup(cola_ecl, tokenizer)

    model_e = ts.AutoModelForSequenceClassification.from_pretrained(
        'google/electra-large-discriminator', cache_dir='tmp/model',
        num_labels=2, classifier_dropout=0.1)

    model_ecl = ts.AutoModelForSequenceClassification.from_pretrained(
        'google/electra-large-discriminator', cache_dir='tmp/model',
        num_labels=2, classifier_dropout=0.1)

    training_args_e = ts.TrainingArguments(
        output_dir='tmp/model_output_e',
        # Batch size args
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        # Optimizer args
        learning_rate=learning_rate_e,
        weight_decay=0,
        # Eval args
        metric_for_best_model='eval_accuracy',
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        # WANDB args
        report_to="wandb",  # enable logging to W&B
    )

    training_args_ecl = ts.TrainingArguments(
        output_dir='tmp/model_output_ecl',
        # Batch size args
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=val_batch_size,
        # Optimizer args
        learning_rate=learning_rate_ecl,
        weight_decay=0,
        # Eval args
        metric_for_best_model='eval_accuracy',
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        # WANDB args
        report_to="wandb",  # enable logging to W&B
    )

    trainer_e = ts.Trainer(
        model=model_e,
        args=training_args_e,
        data_collator=data_collator,
        train_dataset=tokenized_train_e,
        eval_dataset=tokenized_val_e,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    trainer_ecl = ts.Trainer(
        model=model_ecl,
        args=training_args_ecl,
        data_collator=data_collator,
        train_dataset=tokenized_train_ecl,
        eval_dataset=tokenized_val_ecl,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    print('Starting training.')
    # trainer_e.train()
    # trainer_ecl.train()
    print('Finished training.')

    ### Here predictions for summeval and newsroom
    print('Starting scoring for SummEval.')
    summeval_scorer = Scorer('data/SummEval/data.pkl', device, True)
    summeval_scorer.score(['electra_score_e_new'], model_e, tokenizer)
    summeval_scorer.score(['electra_score_ecl_new'], model_ecl, tokenizer)
    summeval_scorer.save_data(f'data/SummEval/{summeval_output}')
    print('Finished scoring for SummEval. \
          Saved results to data/SummEval/{summeval_output}')

    print('Starting scoring for Newsroom.')
    newsroom_scorer = Scorer('data/Newsroom/data.pkl', device, False)
    newsroom_scorer.score(['electra_score_e_new'], model_e, tokenizer)
    newsroom_scorer.score(['electra_score_ecl_new'], model_ecl, tokenizer)
    newsroom_scorer.save_data(f'data/Newsroom/{newsroom_output}')
    print(f'Finished scoring for Newsroom. \
          Saved results to data/Newsroom/{newsroom_output}')

    ### Here evaluation for summeval, newsroom, and newsroom>=4
    print(f"Evaluation of SummEval dataset (data/SummEval/{summeval_output}):")
    summ_stat_summeval = SUMStat(f'data/SummEval/{summeval_output}')
    summ_stat_summeval.evaluate_summary('fluency')
    print(f"Evaluation of Newsroom dataset (data/Newsroom/{newsroom_output}):")
    summ_stat_newsroom = SUMStat(f'data/Newsroom/{newsroom_output}')
    summ_stat_newsroom.evaluate_summary('fluency')
    print(f"Evaluation of Newsroom dataset (data/Newsroom/{newsroom_output})")
    print('Binary casting enabeled with border < 4 + eps')
    summ_stat_newsroom.evaluate_summary(
        'fluency', binary_casting=True, cast_border=4.0+1e-3)
