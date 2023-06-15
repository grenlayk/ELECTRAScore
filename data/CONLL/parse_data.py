import pickle
from collections import Counter


def clean_str(text: str):
    return text.strip().replace('\n', ' ')


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def add_item(jfleg, idx, src_text, text, value):
    jfleg[f'{idx}'] = {
        'src': None,
        'ref_summ': None,
        'sys_summs': {
             f'CONLL': {
                 'sys_summ': None,
                 'scores': {'fluency': None}
             }
         }
    }
    jfleg[f'{idx}']['src'] = src_text
    jfleg[f'{idx}']['ref_summ'] = text
    jfleg[f'{idx}']['sys_summs'][f'CONLL']['sys_summ'] = text
    jfleg[f'{idx}']['sys_summs'][f'CONLL']['scores']['fluency'] = value


def count_noop(a_list):
    count = 0
    for line in a_list:
        if line.find('noop') != -1:
            count += 1
    return count


texts = []
with open('conll14st-test-data/alt/official-2014.combined-withalt.m2', 'r') as f:
    texts = f.readlines()

grouped_text = []
group = []
for line in texts:
    if line[0] == 'S':
        if len(group) > 0:
            grouped_text.append(group)
        group = [clean_str(line[1:])]
    elif line[0] == 'A':
        group.append(clean_str(line))

final_texts = []
final_values = []
for group in grouped_text:
    if len(group) == 1:
        final_texts.append(clean_str(group[0]))
        final_values.append(1.)
    else:
        count = count_noop(group[1:])
        if count == 0 or count == len(group[1:]):
            final_texts.append(clean_str(group[0]))
            final_values.append(0.)

jfleg = dict()
for j, (text, value) in enumerate(zip(final_texts, final_values)):
    add_item(jfleg, j, text, text, value)

save_pickle(jfleg, 'data.pkl')
