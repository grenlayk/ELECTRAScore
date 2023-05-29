# Datasets info

In our work we used 3 datasets: Newsroom, SummEval and CoLA.

**Newsroom dataset**

- Human evaluation data can be downloaded from [official GitHub repo](https://github.com/lil-lab/newsroom/blob/master/humaneval/newsroom-human-eval.csv). 

- Alternative way is usage of BARTScore's `data.pkl` from [Newsroom folder](https://github.com/neulab/BARTScore/tree/main/SUM/Newsroom).

**SummEval dataset**

- Human annotations for summaries can be downloaded from [official GitHub repo](https://github.com/Yale-LILY/SummEval#human-annotations) as well.

- Alternative link to [HuggingFace page](https://huggingface.co/datasets/mteb/summeval).

- Another alternative is usage of BARTScore's `data.pkl` from [SummEval folder](https://github.com/neulab/BARTScore/tree/main/SUM/SummEval).

**CoLA dataset**

- Data can be downloaded from official [web page](https://nyu-mll.github.io/CoLA/). 

- Alternative link to [HuggingFace page](https://huggingface.co/datasets/glue/viewer/cola/train):

## Used data

We decided to use datasets from [BARTScore](https://github.com/neulab/BARTScore) folders (`SUM/SummEval/data.pkl`, `SUM/Newsroom/data.pkl`) which were placed in `data` folder (`data/SummEval/data.pkl` and `data/Newsroom/data.pkl`).

For each dataset, BARTScore's authors converted the original dataset into an unified form as shown below (all texts tokenized, normal cased). The unified data form is in each dataset folder, and is named as `data.pkl`. Note that there is another file `all_metrics_final.pkl` in each dataset folder, which is our calculated score file.

```json
{
    "doc_id": {
        "src": "This is the source text.",
        "ref_summ": "This is the reference summary",
        "sys_summs": {
            "sys_name1": {
                "sys_summ": "This is the system summary.",
                "scores": {
                    "human_metric1": 0.3,
                    "human_metric2": 0.5
                }
            }
        }
    }
}
```

For SummEval dataset, there are multiple references, there is a field called `ref_summs` as shown below. The average is taken when combining multi-reference results.

```json
"ref_summs": [
    "This is the first reference summary.",
    "This is the second reference summary.",
    "..."
]
```

After calculating scores using automatic metrics, the `scores` field for each document is updated, like the one below.
```json
"scores": {
    "auto_metric1": 0.9,
    "auto_metric2": 0.7,
    "human_metric1": 0.3,
    "human_metric2": 0.5
}
```