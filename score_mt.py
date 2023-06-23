""" This code was inherited from BARTScore repo and modified for our needs.
https://github.com/neulab/BARTScore/blob/main/WMT/score.py
Thanks to BARTScore's authors for their work.
"""
import argparse
import os
import time
import numpy as np
from utils import *
from tqdm import tqdm
import torch
from datetime import datetime


class MTScorer:
    """ Support BERTScore, BARTScore, ELECTRAScore """

    def __init__(self, file_path, device='cuda:0'):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.device = device
        self.data = read_pickle(file_path)
        print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
              f'Data loaded from {file_path}.')

        self.refs, self.betters, self.worses = [], [], []
        for doc_id in self.data:
            self.refs.append(self.data[doc_id]['ref'])
            self.betters.append(self.data[doc_id]['better']['sys'])
            if 'worse' in self.data[doc_id]:
                self.worses.append(self.data[doc_id]['worse']['sys'])

    def save_data(self, path):
        save_pickle(self.data, path)

    def record(self, scores_better, scores_worse, name):
        """ Record the scores from a metric """
        for idx, doc_id in enumerate(self.data):
            self.data[doc_id]['better']['scores'][name] = str(
                scores_better[idx])
            if len(self.worses) > 0:
                self.data[doc_id]['worse']['scores'][name] = str(
                    scores_worse[idx])

    def score(self, metrics, model=None, tokenizer=None):
        for metric_name in metrics:
            if metric_name == 'bert_score':
                import bert_score

                def run_bertscore(mt: list, ref: list):
                    """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
                    _, _, f1 = bert_score.score(
                        cands=mt,
                        refs=ref,
                        idf=False,
                        batch_size=32,
                        lang='en',
                        rescale_with_baseline=False,
                        verbose=True,
                        nthreads=4,
                    )
                    return f1.numpy()

                start = time.time()
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
                      'Begin calculating BERTScore.', sep=' ')
                scores_better = run_bertscore(self.betters, self.refs)
                scores_worse = []
                if len(self.worses) > 0:
                    scores_worse = run_bertscore(self.worses, self.refs)
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
                      'Finished calculating BERTScore,',
                      f'time passed {time.time() - start}s.', sep=' ')
                self.record(scores_better, scores_worse, 'bert_score')

            elif (metric_name == 'bart_score' or metric_name == 'bart_score_cnn' or
                  metric_name == 'bart_score_para' or metric_name == 'mbart_score'):
                from metrics.bart_score import BARTScorer

                def run_bartscore(scorer, mt: list, ref: list):
                    hypo_ref = np.array(scorer.score(mt, ref, batch_size=4))
                    ref_hypo = np.array(scorer.score(ref, mt, batch_size=4))
                    avg_f = 0.5 * (ref_hypo + hypo_ref)
                    return avg_f

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(
                        device=self.device,
                        checkpoint='facebook/bart-large-cnn')
                elif 'para' in metric_name:
                    bart_scorer = BARTScorer(
                        device=self.device,
                        checkpoint='facebook/bart-large-cnn')
                    bart_scorer.load()
                else:
                    if 'm' in metric_name:
                        bart_scorer = BARTScorer(
                            device=self.device,
                            checkpoint='facebook/mbart-large-50')
                    else:
                        bart_scorer = BARTScorer(
                            device=self.device,
                            checkpoint='facebook/bart-large')

                start = time.time()
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
                      'Begin calculating BARTScore ({metric_name}).', sep=' ')
                scores_better = run_bartscore(
                    bart_scorer, self.betters, self.refs)
                scores_worse = []
                if len(self.worses) > 0:
                    scores_worse = run_bartscore(
                        bart_scorer, self.worses, self.refs)
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
                      'Finished calculating BARTScore,',
                      'time passed {time.time() - start}s.', sep=' ')
                self.record(scores_better, scores_worse, metric_name)

            elif metric_name == 'bleurt':
                from bleurt import score

                def run_bleurt(
                        candidates: list, references: list, checkpoint: str = "bleurt/BLEURT-20"
                ):
                    scorer = score.BleurtScorer(checkpoint)
                    scores = scorer.score(references=references, candidates=candidates)
                    return scores

                start = time.time()
                print(f'Begin calculating BLEURT.')
                scores_better = run_bleurt(self.betters, self.refs)
                if len(self.worses) > 0:
                    scores_worse = run_bleurt(self.worses, self.refs)
                print(f'Finished calculating BLEURT, time passed {time.time() - start}s.')
                self.record(scores_better, scores_worse, 'bleurt')

            elif (metric_name == 'electra_score' or
                  metric_name[:13] == 'electra_score'):
                """ ELECTRAScore """
                from metrics.electra_score import ELECTRAScorer

                def run_electrascore(scorer, mt: list, sent_agg_func=None):
                    return scorer.score(mt, sent_agg_func=sent_agg_func)

                if metric_name == 'electra_score':
                    electra_scorer = ELECTRAScorer(
                        checkpoint="grenlayk/electra-large-cola",
                        device=self.device)
                else:
                    electra_scorer = ELECTRAScorer(
                        checkpoint=None,
                        model=model,
                        tokenizer=tokenizer,
                        device=self.device)

                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
                      f'ELECTRAScorer for {metric_name} setup finished.',
                      'Begin calculating ELECTRAScore.', sep=' ')

                start = time.time()
                scores_better = run_electrascore(electra_scorer, self.betters)
                scores_better_mean = run_electrascore(
                    electra_scorer, self.betters, sent_agg_func=torch.mean)
                scores_better_min = run_electrascore(
                    electra_scorer, self.betters, sent_agg_func=torch.min)
                scores_better_median = run_electrascore(
                    electra_scorer, self.betters, sent_agg_func=torch.median)

                scores_worse, scores_worse_mean, scores_worse_median, scores_worse_min = [], [], [], []
                if len(self.worses) > 0:
                    scores_worse = run_electrascore(
                        electra_scorer, self.worses)
                    scores_worse_mean = run_electrascore(
                        electra_scorer, self.worses, sent_agg_func=torch.mean)
                    scores_worse_min = run_electrascore(
                        electra_scorer, self.worses, sent_agg_func=torch.min)
                    scores_worse_median = run_electrascore(
                        electra_scorer, self.worses, sent_agg_func=torch.median)
                print(f'[{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}]',
                      'Finished calculating BARTScore,',
                      f'time passed {time.time() - start}s.', sep=' ')
                self.record(scores_better, scores_worse, f'{metric_name}')
                self.record(scores_better_mean,
                            scores_worse_mean, f'{metric_name}_mean')
                self.record(scores_better_median,
                            scores_worse_median, f'{metric_name}_median')
                self.record(scores_better_min,
                            scores_worse_min, f'{metric_name}_min')
            else:
                raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--output', type=str, required=True,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--bert_score', action='store_true', default=False,
                        help='Whether to calculate BERTScore')
    parser.add_argument('--bart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore')
    parser.add_argument('--mbart_score', action='store_true', default=False,
                        help='Whether to calculate BARTScore (multilingual)')
    parser.add_argument('--bart_score_cnn', action='store_true', default=False,
                        help='Whether to calculate BARTScore-CNN')
    parser.add_argument('--bart_score_para', action='store_true', default=False,
                        help='Whether to calculate BARTScore-Para')
    parser.add_argument('--electra_score', action='store_true', default=False,
                        help='Whether to calculate ELECTRAScore')
    parser.add_argument('--bleurt', action='store_true', default=False,
                        help='Whether to calculate ELECTRAScore')
    args = parser.parse_args()

    scorer = MTScorer(args.file, args.device)

    METRICS = []
    if args.bert_score:
        METRICS.append('bert_score')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.mbart_score:
        METRICS.append('mbart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.bart_score_para:
        METRICS.append('bart_score_para')
    if args.electra_score:
        METRICS.append('electra_score')
    if args.bleurt:
        METRICS.append('bleurt')

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()
