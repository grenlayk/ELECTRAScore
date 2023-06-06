# %%
import numpy as np
from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

from utils import read_pickle


class SUMStat:
    """ A class used to get stats of SUM trained data """

    def __init__(self, path):
        self.path = path
        self.data = read_pickle(path)
        self.sample_id = list(self.data.keys())[0]
        self.sample_sys = list(
            self.data[self.sample_id]['sys_summs'].keys())[0]
        self._metrics = list(
            self.data[self.sample_id]['sys_summs'][self.sample_sys]
            ['scores'].keys()
        )
        self._auto_metrics = [x for x in self.metrics
                              if x not in self.human_metrics]

    def evaluate_summary(
        self,
        human_metric='fluency',
        auto_metrics=None,
        table=None,
        binary_casting=False,
        cast_border=None,
    ):
        """ Evaluate summaries.
        Conduct summary-level correlations w.r.t each document. """
        assert human_metric in self.human_metrics
        if auto_metrics is None:
            auto_metrics = self.auto_metrics
        print(f'Human metric: {human_metric}')
        headers = ['metric', 'spearman', 'kendalltau']
        metric_with_corr = []
        for metric in auto_metrics:
            correlations = []
            target_scores = []
            prediction_scores = []

            for doc_id in self.data:
                sys_summs = self.data[doc_id]['sys_summs']
                for sys_name in sys_summs:
                    prediction_scores.append(
                        sys_summs[sys_name]['scores'][metric])
                    target_scores.append(
                        sys_summs[sys_name]['scores'][human_metric])

            if binary_casting:
                border = 4.0
                if cast_border == 'mean':
                    border = np.mean(target_scores)
                elif cast_border == 'median':
                    border = np.median(target_scores)
                target_scores = [0. if score < border else 1. for score in target_scores]

            if (len(set(prediction_scores)) == 1 or
                    len(set(target_scores)) == 1):
                continue

            correlations.append([
                spearmanr(target_scores, prediction_scores)[0],
                kendalltau(target_scores, prediction_scores)[0]
            ])
            # print(correlations[-1])
            corr_mat = np.array(correlations)
            spearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1])
            metric_with_corr.append([metric, spearman, ktau])
        sorted_metric_with_corr = sorted(
            metric_with_corr, key=lambda x: x[1], reverse=True)
        if table is not None:
            file = open(table, 'w') 
            for each in sorted_metric_with_corr:
                print(f'{each[0]}\t{each[1]}\t{each[2]}', file=file)
            file.flush()
        if binary_casting:
            print(' < ', border, '')
        print(tabulate(
            sorted_metric_with_corr, headers=headers, tablefmt='simple'))

    @property
    def auto_metrics(self):
        return self._auto_metrics

    @property
    def metrics(self):
        return self._metrics

    @property
    def human_metrics(self):
        """ All available human metrics. """
        if 'SummEval' in self.path:
            return ['coherence', 'consistency', 'fluency', 'relevance']
        if 'Newsroom' in self.path:
            return ['coherence', 'fluency', 'informativeness', 'relevance']
