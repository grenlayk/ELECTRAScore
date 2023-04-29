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
bert_score_r              0.190092     0.153538
bart_score_cnn_avg_f      0.13216      0.110898
bart_score_cnn_harm_f     0.126831     0.105637
bert_score_p              0.120363     0.097894
bart_score_cnn_hypo_ref   0.113409     0.0913985
bart_score_cnn_ref_hypo   0.109333     0.0910405
bart_score_harm_f         0.0379608    0.0292604
bart_score_ref_hypo       0.0128045    0.0110808
bart_score_hypo_ref       0.00176333   0.000597686
bart_score_avg_f         -0.00436448  -0.001944
```

Rouge
```
Human metric: fluency
metric      spearman    kendalltau
--------  ----------  ------------
rouge2_f   0.158797      0.128203
rouge2_p   0.137032      0.111975
rouge1_f   0.114974      0.0938505
rouge2_r   0.11303       0.0921418
rougel_f   0.104976      0.0841752
rouge1_p   0.0947817     0.0803175
rougel_p   0.0832704     0.0704288
rougel_r   0.0827189     0.0692879
rouge1_r   0.0744546     0.0608603
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
bart_score_cnn_ref_hypo   0.352645     0.286757
bart_score_cnn_avg_f      0.347317     0.285245
bart_score_cnn_harm_f     0.333977     0.273559
bert_score_r              0.292612     0.231461
bart_score_hypo_ref       0.291634     0.226363
bart_score_cnn_hypo_ref   0.235044     0.183404
bart_score_harm_f         0.202404     0.163598
bert_score_f              0.140051     0.108461
bert_score_p              0.0389706    0.0409785
bart_score_avg_f         -0.0128456   -0.00872257
bart_score_ref_hypo      -0.214758    -0.171584
```

Rouge
```
Human metric: fluency
metric       spearman    kendalltau
--------  -----------  ------------
rouge1_r   0.220181      0.187634
rougel_r   0.199403      0.176571
rouge1_f   0.103553      0.0818111
rouge2_r   0.0927091     0.0743487
rougel_f   0.0646335     0.0552827
rouge2_p   0.0479652     0.0338056
rouge2_f   0.0478407     0.0324639
rougel_p  -0.00314919    0.00826815
rouge1_p  -0.00755879   -0.00200908
```

All metrics (except rouge): https://pastebin.com/rwjYFnfp

## Issues

### Rouge installation & execution

Tried many thing (e.g. install and reinstall, but failed). Decided to resolve later.

Links collected during attempts:
- https://gist.github.com/donglixp/d7eea02d57ba2e099746f8463c2f6597
- https://medium.com/@prabha88978/installation-working-process-of-rouge-1-5-5-6c0dfdca49e8 

TODO: try to reinstall and do this https://github.com/neulab/BARTScore/issues/17.



