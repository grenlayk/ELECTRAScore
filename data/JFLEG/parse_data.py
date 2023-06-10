import pickle
import itertools
from collections import defaultdict
import numpy as np


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def clean_str(text: str):
    return text.strip().replace('\n', ' ')


def add_item(id, i, text, value):
    jfleg[f'{id}_{i}'] = {
        'src': None,
        'ref_summ': None,
        'sys_summs': {
             f'JFLEG': {
                 'sys_summ': None,
                 'scores': {'fluency': None}
             }
         }
    }
    jfleg[f'{id}_{i}']['src'] = text
    jfleg[f'{id}_{i}']['ref_summ'] = text
    jfleg[f'{id}_{i}']['sys_summs'][f'JFLEG']['sys_summ'] = text
    jfleg[f'{id}_{i}']['sys_summs'][f'JFLEG']['scores']['fluency'] = value
    return value


ref_num = 4
dev = []
test = []

for i in range(ref_num):
    with open(f'jfleg/dev/dev.ref{i}', 'r') as f:
        dev.append(list(map(clean_str, f.readlines())))

for i in range(ref_num):
    with open(f'jfleg/test/test.ref{i}', 'r') as f:
        test.append(list(map(clean_str, f.readlines())))

with open(f'jfleg/dev/dev.spellchecked.src', 'r') as f:
    dev.append(list(map(clean_str, f.readlines())))

with open(f'jfleg/test/test.spellchecked.src', 'r') as f:
    test.append(list(map(clean_str, f.readlines())))


dev = np.array(dev).flatten('F')
test = np.array(test).flatten('F')

dev_values = [1. if i % 5 != 4 else 0. for i in range(len(dev))]
test_values = [1. if i % 5 != 4 else 0. for i in range(len(test))]

for i in range(0, len(dev), 5):
    if dev[i+4] in set(dev[i:i+4]):
        dev_values[i+4] = 1.

for i in range(0, len(test), 5):
    if test[i+4] in set(test[i:i+4]):
        test_values[i+4] = 1.

jfleg = dict()

for id, ref, values in zip(['dev', 'test'], [dev, test], [dev_values, test_values]):
    for i, text in enumerate(ref):
        add_item(id, i, text, value=values[i])

save_pickle(jfleg, 'data.pkl')
