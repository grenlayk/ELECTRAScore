# %%
import numpy as np
from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

from utils import read_pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def calculate_precision_recall_auc(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return pr_auc


def custom_score(target_scores, prediction_scores):
    if (prediction_scores.index(min(prediction_scores)) ==
            target_scores.index(min(target_scores))):
        return 1
    return 0


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

    def evaluate(
        self,
        human_metric='fluency',
        auto_metrics=None,
        table=None,
        dataset_level=False,
    ):
        assert human_metric in self.human_metrics
        if auto_metrics is None:
            auto_metrics = self.auto_metrics
        print(f'Human metric: {human_metric}')
        headers = ['metric', 'roc_auc', 'pr_auc']
        if not dataset_level:
            headers.append('student < teacher')

        metric_with_score = []
        for metric in auto_metrics:
            metrics = []
            target_scores = []
            prediction_scores = []
            block_id = 0
            for doc_id in self.data:
                if not dataset_level and int(doc_id[:doc_id.find('_')]) > block_id:
                    metrics.append([
                        roc_auc_score(target_scores, prediction_scores),
                        calculate_precision_recall_auc(
                            target_scores, prediction_scores),
                        custom_score(target_scores, prediction_scores)])
                    target_scores = []
                    prediction_scores = []
                    block_id += 1
                sys_summs = self.data[doc_id]['sys_summs']
                for sys_name in sys_summs:
                    prediction_scores.append(
                        sys_summs[sys_name]['scores'][metric])
                    target_scores.append(
                        sys_summs[sys_name]['scores'][human_metric])

            if dataset_level:
                metrics.append([
                    roc_auc_score(target_scores, prediction_scores),
                    calculate_precision_recall_auc(
                        target_scores, prediction_scores),
                ])
            else:
                metrics.append([
                    roc_auc_score(target_scores, prediction_scores),
                    calculate_precision_recall_auc(
                        target_scores, prediction_scores),
                    custom_score(target_scores, prediction_scores)])

            metrics_mat = np.array(metrics)
            roc_auc, pr_auc = np.mean(metrics_mat[:, 0]), np.mean(metrics_mat[:, 1])

            if not dataset_level:
                custom = np.mean(metrics_mat[:, 2])
                metric_with_score.append([metric, roc_auc, pr_auc, custom])
            else:
                metric_with_score.append([metric, roc_auc, pr_auc])
        sorted_metric_with_corr = sorted(
            metric_with_score, key=lambda x: x[1], reverse=True)
        if table is not None:
            file = open(table, 'w')
            for each in sorted_metric_with_corr:
                print(f'{each[0]}\t{each[1]}\t{each[2]}', file=file)
            file.flush()
        print(tabulate(
            sorted_metric_with_corr, headers=headers, tablefmt='simple'))

    def evaluate_summary(
        self,
        human_metric='fluency',
        auto_metrics=None,
        table=None,
        dataset_level=False,
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
                if not dataset_level:
                    if len(set(prediction_scores)) == 1 or len(set(target_scores)) == 1:
                        continue
                    correlations.append([
                        spearmanr(target_scores, prediction_scores)[0],
                        kendalltau(target_scores, prediction_scores)[0]])
                    target_scores = []
                    prediction_scores = []

            if binary_casting:
                border = cast_border
                if cast_border == 'mean':
                    border = np.mean(target_scores)
                elif cast_border == 'median':
                    border = np.median(target_scores)
                target_scores = [
                    0. if score < border else 1. for score in target_scores]

            if dataset_level:
                if (len(set(prediction_scores)) == 1 or
                        len(set(target_scores)) == 1):
                    continue

                correlations.append([
                    spearmanr(target_scores, prediction_scores)[0],
                    kendalltau(target_scores, prediction_scores)[0]
                ])

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
            print(' < ', border)
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
        if 'JFLEG' in self.path:
            return ['fluency']
        if 'CONLL' in self.path:
            return ['fluency']
