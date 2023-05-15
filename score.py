import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import detokenize, read_pickle, save_pickle


class Scorer:
    """ Support BERTScore, BARTScore """

    def __init__(self, file_path, device='cuda:0', multi_ref=False):
        """ file_path: path to the pickle file
            All the data are normal capitalized, and tokenized, including src,
            ref_summ, ref_summs, and sys_summ.
        """
        self.multi_ref = multi_ref
        self.device = device
        self.data = read_pickle(file_path)
        print(f'Data loaded from {file_path}.')

        self.sys_names = self.get_sys_names()

        if not multi_ref:
            self.single_ref_lines = self.get_single_ref_lines()
            print('In a single-reference setting.')
        else:
            self.multi_ref_lines = self.get_multi_ref_lines()
            self.ref_num = len(self.multi_ref_lines[0])
            print('In a multi-reference setting.')

    def get_sys_names(self):
        first_id = list(self.data.keys())[0]
        return list(self.data[first_id]['sys_summs'].keys())

    def get_single_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summ'])
        return ref_lines

    def get_multi_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summs'])
        return ref_lines

    def get_sys_lines(self, sys_name):
        sys_lines = []
        for doc_id in self.data:
            sys_lines.append(
                self.data[doc_id]['sys_summs'][sys_name]['sys_summ'])
        return sys_lines

    def get_src_lines(self):
        src_lines = []
        for doc_id in self.data:
            src_lines.append(self.data[doc_id]['src'])
        return src_lines

    def save_data(self, path):
        save_pickle(self.data, path)

    def score(self, metrics):
        """ metrics: list of metrics """
        for metric_name in metrics:
            if metric_name == 'bert_score':
                from bert_score import BERTScorer

                # Set up BERTScore
                bert_scorer = BERTScorer(
                    lang='en',
                    idf=False,
                    rescale_with_baseline=True,
                    device=self.device
                )
                print('BERTScore setup finished. Begin calculating BERTScore.')

                start = time.time()
                ref_lines = (self.single_ref_lines if not self.multi_ref else
                             self.multi_ref_lines)
                for sys_name in tqdm(self.sys_names):
                    sys_lines = self.get_sys_lines(sys_name)
                    if not self.multi_ref:
                        P, R, F = bert_scorer.score(sys_lines, ref_lines)
                    else:
                        total_num = len(sys_lines)
                        P, R, F = (np.zeros(total_num), np.zeros(total_num),
                                   np.zeros(total_num))
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_P, curr_R, curr_F = bert_scorer.score(
                                sys_lines, ref_list)
                            P += curr_P.numpy()
                            R += curr_R.numpy()
                            F += curr_F.numpy()
                        P, R, F = (P / self.ref_num, R / self.ref_num,
                                   F / self.ref_num)
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name][
                            'scores'].update({
                                # 'bert_score_p': P[counter],
                                # 'bert_score_r': R[counter],
                                'bert_score_f': F[counter]
                            })
                        counter += 1
                print(f'Finished calculating BERTScore, time passed \
                      {time.time() - start}s.')

            elif (metric_name == 'electra_score' or
                  metric_name == 'electra_score_extended' or
                  metric_name == 'electra_score_extended_chatgpt'):
                """ Vanilla ELECTRAScore """
                from metrics.electra_score import ELECTRAScorer

                if metric_name == 'electra_score':
                    electra_scorer = ELECTRAScorer(
                        checkpoint="grenlayk/electra-large-cola",
                        device=self.device)
                elif metric_name == 'electra_score_extended':
                    electra_scorer = ELECTRAScorer(
                        checkpoint="grenlayk/electra-large-cola-extended",
                        device=self.device)
                else:
                    electra_scorer = ELECTRAScorer(
                        checkpoint="grenlayk/electra-large-cola-extended-chatgpt",
                        device=self.device)

                print(f'ELECTRAScorer for {metric_name} setup finished. \
                      Begin calculating ELECTRAScore.')

                start = time.time()

                for sys_name in tqdm(self.sys_names):
                    sys_lines = self.get_sys_lines(sys_name)
                    scores = electra_scorer.score(sys_lines)
                    scores_mean = electra_scorer.score(
                        sys_lines, sent_agg_func=torch.mean)
                    scores_min = electra_scorer.score(
                        sys_lines, sent_agg_func=torch.min)
                    scores_median = electra_scorer.score(
                        sys_lines, sent_agg_func=torch.median)

                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name][
                            'scores'].update({
                                f'{metric_name}': scores[counter],
                                f'{metric_name}_mean': scores_mean[counter],
                                f'{metric_name}_min': scores_min[counter],
                                f'{metric_name}_median':
                                    scores_median[counter],
                            })
                        counter += 1
                print(f'Finished calculating ELECTRAScore ({metric_name}), \
                      time passed {time.time() - start}s.')

            elif (metric_name == 'bart_score' or
                  metric_name == 'bart_score_cnn'):
                """ Vanilla BARTScore, BARTScore-CNN """
                from metrics.bart_score import BARTScorer

                # Set up BARTScore
                if 'cnn' in metric_name:
                    bart_scorer = BARTScorer(
                        device=self.device,
                        checkpoint='facebook/bart-large-cnn')
                else:
                    bart_scorer = BARTScorer(
                        device=self.device, checkpoint='facebook/bart-large')
                print('BARTScore setup finished. Begin calculating BARTScore.')

                start = time.time()
                # Keep capitalization, detokenize everything
                src_lines = self.get_src_lines()
                src_lines = [detokenize(line) for line in src_lines]
                if not self.multi_ref:
                    ref_lines = [
                        detokenize(line) for line in self.single_ref_lines]
                else:
                    ref_lines = [[detokenize(text) for text in line]
                                 for line in self.multi_ref_lines]
                for sys_name in tqdm(self.sys_names):
                    sys_lines = self.get_sys_lines(sys_name)
                    sys_lines = [detokenize(line) for line in sys_lines]
                    src_hypo = bart_scorer.score(
                        src_lines, sys_lines, batch_size=4)
                    if not self.multi_ref:
                        ref_hypo = np.array(
                            bart_scorer.score(
                                ref_lines, sys_lines, batch_size=4))
                        hypo_ref = np.array(
                            bart_scorer.score(
                                sys_lines, ref_lines, batch_size=4))
                    else:
                        ref_hypo, hypo_ref = (np.zeros(len(sys_lines)),
                                              np.zeros(len(sys_lines)))
                        for i in range(self.ref_num):
                            ref_list = [x[i] for x in ref_lines]
                            curr_ref_hypo = np.array(bart_scorer.score(
                                ref_list, sys_lines, batch_size=4))
                            curr_hypo_ref = np.array(bart_scorer.score(
                                sys_lines, ref_list, batch_size=4))
                            ref_hypo += curr_ref_hypo
                            hypo_ref += curr_hypo_ref
                        ref_hypo = ref_hypo / self.ref_num
                        hypo_ref = hypo_ref / self.ref_num
                    avg_f = (ref_hypo + hypo_ref) / 2
                    harm_f = (ref_hypo * hypo_ref) / (ref_hypo + hypo_ref)
                    counter = 0
                    for doc_id in self.data:
                        self.data[doc_id]['sys_summs'][sys_name][
                            'scores'].update({
                                f'{metric_name}_src_hypo': src_hypo[counter],
                                # f'{metric_name}_hypo_ref': hypo_ref[counter],
                                # f'{metric_name}_ref_hypo': ref_hypo[counter],
                                # f'{metric_name}_avg_f': avg_f[counter],
                                # f'{metric_name}_harm_f': harm_f[counter]
                            })
                        counter += 1
                print(f'Finished calculating BARTScore, \
                      time passed {time.time() - start}s.')

            else:
                raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument(
        '--file', type=str, required=True, help='The data to load from.')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='The device to run on.')
    parser.add_argument(
        '--multi_ref', action='store_true', default=False,
        help='Whether we are using multiple references to calculate scores.')
    parser.add_argument(
        '--output', type=str, required=True,
        help='The output path to save the calculated scores.')
    parser.add_argument(
        '--bert_score', action='store_true', default=False,
        help='Whether to calculate BERTScore')
    parser.add_argument(
        '--electra_score', action='store_true', default=False,
        help='Whether to calculate ELECTRAScore')
    parser.add_argument(
        '--electra_score_extended', action='store_true', default=False,
        help='Whether to calculate ELECTRAScore on extended CoLA')
    parser.add_argument(
        '--electra_score_extended_chatgpt', action='store_true', default=False,
        help='Whether to calculate ELECTRAScore on extended CoLA with ChatGPT \
            labels')
    parser.add_argument(
        '--bart_score', action='store_true', default=False,
        help='Whether to calculate BARTScore')
    parser.add_argument(
        '--bart_score_cnn', action='store_true', default=False,
        help='Whether to calculate BARTScore-CNN')
    args = parser.parse_args()

    scorer = Scorer(args.file, args.device, args.multi_ref)

    METRICS = []
    if args.bert_score:
        METRICS.append('bert_score')
    if args.bart_score:
        METRICS.append('bart_score')
    if args.bart_score_cnn:
        METRICS.append('bart_score_cnn')
    if args.electra_score:
        METRICS.append('electra_score')
    if args.electra_score_extended:
        METRICS.append('electra_score_extended')
    if args.electra_score_extended_chatgpt:
        METRICS.append('electra_score_extended_chatgpt')

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()
