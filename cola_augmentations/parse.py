"""
Original file can be found here:
https://github.com/dheeraj7596/CONDA/blob/main/combine.py

The file was modified in order to add instructions for CoLA dataset.
We removed merging part and modified clean function.
"""

import pandas as pd
from collections import Counter
import sys
import os
import numpy as np


def clean(context):
    temp = context.split("context :")[1:-1]
    temp = [line.split('\n')[0] for line in temp]
    temp = [row.replace('<|endoftext|>', ' ').strip() for row in temp]
    return temp


if __name__ == "__main__":
    base_path = sys.argv[1]
    gen_file = sys.argv[2]

    df_gen = pd.read_csv(gen_file)

    texts = []
    labels = []
    ids = []
    for i, row in df_gen.iterrows():
        context = row["generated_context"]
        clean_context = clean(context)
        if clean_context is None or len(clean_context) == 0:
            continue
        texts.extend(clean_context)
        labels.extend(
            [1 if row["label"] == 'positive' else 0] * len(clean_context))
        ids.extend(["gen_" + str(i)] * len(clean_context))

    dict_df = pd.DataFrame.from_dict(
        {"text": texts, "label": labels, "id": ids})

    dict_df["text"].replace("", np.nan, inplace=True)
    dict_df = dict_df.dropna(subset=["text"]).reset_index(drop=True)
    dict_df = dict_df.drop_duplicates(subset=['text', 'label'])
    print(Counter(dict_df["label"]))
    dict_df.to_csv(os.path.join(base_path, "generated_data.csv"), index=False)
