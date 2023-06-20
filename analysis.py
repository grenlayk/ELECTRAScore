# %%
import numpy as np
from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

from utils import read_pickle, save_pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import random
from tqdm import tqdm


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


class WMTStat:
    """ A class used to get stats of WMT trained data """

    def __init__(self, path):
        self.path = path
        self.data = read_pickle(path)
        self._metrics = list(self.data[next(iter(self.data))]['better']['scores'].keys())
        pos = path.find('-en')
        self.lp = path[pos - 2: pos + 3]
        self._auto_metrics = [x for x in self.metrics
                              if x not in self.human_metrics]
        # systems ranked by their DA score
        self._systems = {
            'de-en': ['Facebook_FAIR.6750', 'RWTH_Aachen_System.6818', 'MSRA.MADL.6910', 'online-B.0', 'JHU.6809',
                      'MLLP-UPV.6899', 'dfki-nmt.6478', 'UCAM.6461', 'online-A.0', 'NEU.6801', 'uedin.6749',
                      'online-Y.0', 'TartuNLP-c.6502', 'online-G.0', 'PROMT_NMT_DE-EN.6683', 'online-X.0'],
            'fi-en': ['MSRA.NAO.6983', 'online-Y.0', 'GTCOM-Primary.6946', 'USYD.6995', 'online-B.0',
                      'Helsinki_NLP.6889', 'online-A.0', 'online-G.0', 'TartuNLP-c.6905', 'online-X.0', 'parfda.6526',
                      'apertium-fin-eng-unconstrained-fien.6449'],
            'gu-en': ['NEU.6756', 'UEDIN.6534', 'GTCOM-Primary.6969', 'CUNI-T2T-transfer-guen.6431',
                      'aylien_mt_gu-en_multilingual.6826', 'NICT.6603', 'online-G.0', 'IITP-MT.6824', 'UdS-DFKI.6861',
                      'IIITH-MT.6688', 'Ju_Saarland.6525'],
            'kk-en': ['online-B.0', 'NEU.6753', 'rug_kken_morfessor.6677', 'online-G.0', 'talp_upc_2019_kken.6657',
                      'NRC-CNRC.6895', 'Frank_s_MT.6127', 'NICT.6770', 'CUNI-T2T-transfer-kken.6436', 'UMD.6736',
                      'DBMS-KU_KKEN.6726'],
            'lt-en': ['GTCOM-Primary.6998', 'tilde-nc-nmt.6881', 'NEU.6759', 'MSRA.MASS.6945', 'tilde-c-nmt.6876',
                      'online-B.0', 'online-A.0', 'TartuNLP-c.6908', 'online-G.0', 'JUMT.6616', 'online-X.0'],
            'ru-en': ['Facebook_FAIR.6937', 'online-G.0', 'eTranslation.6598', 'online-B.0', 'NEU.6803',
                      'MSRA.SCA.6976', 'rerank-re.6540', 'online-Y.0', 'online-A.0', 'afrl-syscomb19.6782',
                      'afrl-ewc.6659', 'TartuNLP-u.6650', 'online-X.0', 'NICT.6561'],
            'zh-en': ['Baidu-system.6940', 'KSAI-system.6927', 'MSRA.MASS.6996', 'MSRA.MASS.6942', 'NEU.6832',
                      'BTRANS.6825', 'online-B.0', 'BTRANS-ensemble.6992', 'UEDIN.6530', 'online-Y.0', 'NICT.6814',
                      'online-A.0', 'online-G.0', 'online-X.0', 'Apprentice-c.6706']
        }

    def save_data(self, path=None):
        if path is None:
            path = self.path
        save_pickle(self.data, path)

    def retrieve_scores(self, metric, doc_ids):
        """ retrieve better, worse scores """
        better, worse = [], []
        for doc_id in doc_ids:
            better.append(float(self.data[doc_id]['better']['scores'][metric]))
            if 'worse' in self.data[doc_id]:
                worse.append(float(self.data[doc_id]['worse']['scores'][metric]))
        return better, worse

    def kendall(self, hyp1_scores: list, hyp2_scores: list):
        """ Computes the official WMT19 shared task Kendall correlation score. """
        assert len(hyp1_scores) == len(hyp2_scores)
        conc, disc = 0, 0

        for x1, x2 in zip(hyp1_scores, hyp2_scores):
            if x1 > x2:
                conc += 1
            else:
                disc += 1
        return (conc - disc) / (conc + disc)

    def print_ktau(self, metrics=None):
        headers = ['metric', 'k-tau']
        metric_with_ktau = []
        doc_ids = list(self.data.keys())
        if metrics is None:
            metrics = self.metrics
        for metric in tqdm(metrics):
            better, worse = self.retrieve_scores(metric, doc_ids)
            ktau = self.kendall(better, worse)
            metric_with_ktau.append([metric, ktau])
        sorted_metric_with_ktau = sorted(metric_with_ktau, key=lambda x: x[1], reverse=True)
        print(tabulate(sorted_metric_with_ktau, headers=headers, tablefmt='simple'))

    def print_ref_len(self):
        """ Get the length of reference texts """
        ref_lens = []
        for doc_id in self.data:
            ref = self.data[doc_id]['ref']
            ref_len = len(ref.split(' '))
            ref_lens.append(ref_len)
        print(f'Mean reference length: {np.mean(ref_lens)}')
        print(f'Max reference length: {np.max(ref_lens)}')
        print(f'Min reference length: {np.min(ref_lens)}')
        print(f'20% percentile: {np.percentile(ref_lens, 20)}')
        print(f'80% percentile: {np.percentile(ref_lens, 80)}')
        print(f'90% percentile: {np.percentile(ref_lens, 90)}')

    def print_len_ktau(self, min_len, max_len, metrics=None):
        headers = ['metric', 'k-tau']
        metric_with_ktau = []
        sub_ids = []
        for doc_id in tqdm(self.data):
            ref_len = len(self.data[doc_id]['ref'].split(' '))
            if min_len <= ref_len <= max_len:
                sub_ids.append(doc_id)
        print(f'Considered samples: {len(sub_ids)}')
        if metrics is None:
            metrics = self.metrics
        for metric in tqdm(metrics):
            better, worse = self.retrieve_scores(metric, sub_ids)
            ktau = self.kendall(better, worse)
            metric_with_ktau.append([metric, ktau])
        sorted_metric_with_ktau = sorted(metric_with_ktau, key=lambda x: x[1], reverse=True)
        print(tabulate(sorted_metric_with_ktau, headers=headers, tablefmt='simple'))

    def evaluate_translation(
        self,
        human_metric='fluency',
        auto_metrics=None,
        table=None,
    ):
        """ Evaluate translations. """
        if auto_metrics is None:
            auto_metrics = self.auto_metrics
        print(f'Human metric: {human_metric}')
        headers = ['metric', 'roc_auc', 'pr_auc']
        metric_with_corr = []
        for metric in auto_metrics:
            correlations = []
            target_scores = []
            prediction_scores = []
            for doc_id in self.data:
                prediction_scores.append(float(self.data[doc_id]['better']['scores'][metric]))
                target_scores.append(float(self.data[doc_id]['better']['scores'][human_metric]))
                if 'worse' in self.data[doc_id]:
                    prediction_scores.append(float(self.data[doc_id]['worse']['scores'][metric]))
                    target_scores.append(float(self.data[doc_id]['worse']['scores'][human_metric]))

            if (len(set(prediction_scores)) == 1 or
                    len(set(target_scores)) == 1):
                continue

            correlations.append([
                roc_auc_score(target_scores, prediction_scores),
                calculate_precision_recall_auc(target_scores, prediction_scores)
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
        print(tabulate(
            sorted_metric_with_corr, headers=headers, tablefmt='simple'))

    @property
    def metrics(self):
        return self._metrics

    @property
    def systems(self):
        return self._systems[self.lp]

    @property
    def auto_metrics(self):
        return self._auto_metrics

    @property
    def human_metrics(self):
        """ All available human metrics. """
        if 'WMT' in self.path:
            return ['fluency']