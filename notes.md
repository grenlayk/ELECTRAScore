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
```

Rouge
```
Human metric: fluency
metric      spearman    kendalltau
--------  ----------  ------------
rouge2_f   0.158797      0.128203
rouge1_f   0.114974      0.0938505
rougel_f   0.104976      0.0841752
```

ELECTRAScore
```
Human metric: fluency
metric                  spearman    kendalltau
--------------------  ----------  ------------
electra_score           0.388951      0.316606
electra_score_mean      0.355723      0.290099
electra_score_min       0.351573      0.293277
electra_score_median    0.296727      0.244721
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
```

Rouge
```
Human metric: fluency
metric       spearman    kendalltau
--------  -----------  ------------
rouge1_f   0.103553      0.0818111
rougel_f   0.0646335     0.0552827
rouge2_f   0.0478407     0.0324639
```

ELECTRAScore
```
Human metric: fluency
metric                  spearman    kendalltau
--------------------  ----------  ------------
electra_score_median    0.491056      0.399968
electra_score_mean      0.433697      0.360391
electra_score           0.429291      0.359955
electra_score_min       0.362858      0.294209
```




