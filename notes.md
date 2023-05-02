# Notes on thesis

This document is created to track thesis's progress, links, sources and etc.

## Datasets

### Newsroom dataset

I downloaded data manually from [official webpage](https://lil.nlp.cornell.edu/newsroom/download/index.html) after signing a licence agreement. 

Alternative link to [HuggingFace page](https://huggingface.co/datasets/newsroom).

```Python
from datasets import load_dataset

dataset = load_dataset("newsroom")
```

Another alternative is usage of BARTScore's `data.pkl` from [Newsroom folder](https://github.com/neulab/BARTScore/tree/main/SUM/Newsroom).

### SummEval dataset

Human annotations for summaries were downloaded manually as well, from [official GitHub repo](https://github.com/Yale-LILY/SummEval#human-annotations).

Alternative link to [HuggingFace page](https://huggingface.co/datasets/mteb/summeval).

```Python
from datasets import load_dataset

dataset = load_dataset("mteb/summeval")
```

Another alternative is usage of BARTScore's `data.pkl` from [SummEval folder](https://github.com/neulab/BARTScore/tree/main/SUM/SummEval).

### CoLA dataset

Downloaded manually from official [web page](https://nyu-mll.github.io/CoLA/). However it can be used throught [HuggingFace](https://huggingface.co/datasets/glue/viewer/cola/train):

```Python
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
```

## BARTScore reproduction

I decided to use datasets from [BARTScore](https://github.com/neulab/BARTScore) folders (`SUM/SummEval/data.pkl`, `SUM/Newsroom/data/pkl`).

```bash
# running from BARTScore root folder
cd SUM
mkdir models
gdown 1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m 
mv bart_score.pth models/bart_score.pth # downloaded file should be in models folder
python3 score.py --file SummEval/data.pkl --device mps:0 --output SummEval/scores.pkl --bert_score --rouge --bart_score --bart_score_cnn  --multi_ref
python3 score.py --file Newsroom/data.pkl --device mps:0 --output Newsroom/scores.pkl --bert_score --rouge --bart_score --bart_score_cnn 
```

This code downloads BARTScore model checkpoint and runs model on SummEval and Newsroom datasets, creating `SummEval/scores.pkl` and `Newsroom/scores.pkl` with `rouge`, `bert_score`, `bart_score` and `bart_score_cnn` scores in `"scores"` field.

In order to evaluate metrics for SummEval and calculate spearman correlation score we can use this code:

```Python
summ_stat = SUMStat('SUM/SummEval/scores.pkl') 
summ_stat.evaluate_summary('fluency') 
```

Results:

```
Human metric: fluency
metric                      spearman    kendalltau
-----------------------  -----------  ------------
bart_score_cnn_src_hypo   0.356487     0.292136
bart_score_src_hypo       0.24802      0.202521
bert_score_f              0.193229     0.157773
rouge2_f                  0.158797     0.128203
rouge1_f                  0.114974     0.0938505
rougel_f                  0.104976     0.0841752
```

ELECTRAScore
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score                  0.388793      0.31634
electra_score_mean             0.355717      0.290085
electra_score_min              0.350164      0.291133
electra_score_percentile_25    0.330365      0.271945
electra_score_median           0.296147      0.243935
electra_score_percentile_75    0.243651      0.198214
```

In order to evaluate metrics for Newsroom and calculate spearman correlation score we can use this code:

```Python
summ_stat = SUMStat('SUM/Newsroom/scores.pkl') 
summ_stat.evaluate_summary('fluency') 
```

Results:

```
Human metric: fluency
metric                     spearman    kendalltau
-----------------------  ----------  ------------
bart_score_src_hypo       0.670134     0.563899
bart_score_cnn_src_hypo   0.639777     0.540041
bert_score_f              0.140051     0.108461
rouge1_f                  0.103553     0.081811
rougel_f                  0.064634     0.055283
rouge2_f                  0.047841     0.032464
```

ELECTRAScore
```
Human metric: fluency
metric                         spearman    kendalltau
---------------------------  ----------  ------------
electra_score_median           0.492731      0.401303
electra_score_percentile_75    0.487944      0.395381
electra_score_mean             0.434852      0.361143
electra_score                  0.430446      0.360707
electra_score_percentile_25    0.422495      0.349833
electra_score_min              0.363339      0.29292
```
