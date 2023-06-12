import pickle
import numpy as np


REF_NUM = 4


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def clean_str(text: str):
    return text.strip().replace('\n', ' ')


def add_item(jfleg, id, i, src, refs, values, not_none_id=0):
    refs_dict = dict()
    for j in range(REF_NUM):
        refs_dict[f'ref_{j}'] = {
            'sys_summ': None,
            'scores': {'fluency': None}
        }
    jfleg[f'{id}_{i}'] = {
        'src': None,
        'ref_summ': None,
        'sys_summs': refs_dict,
    }
    jfleg[f'{id}_{i}']['src'] = src
    jfleg[f'{id}_{i}']['ref_summ'] = refs[not_none_id]

    count = 1
    jfleg[f'{id}_{i}']['sys_summs'][f'src'] = {
        'sys_summ': src,
        'scores': {'fluency': values[REF_NUM]},
    }
    for j in range(REF_NUM):
        jfleg[f'{id}_{i}']['sys_summs'][f'ref_{j}']['sys_summ'] = refs[j]
        jfleg[f'{id}_{i}']['sys_summs'][f'ref_{j}']['scores']['fluency'] = values[j]
        if values[j] is not None:
            count += 1
    return count


def create_dataset(
        change_student_values: bool = True,
        remove_duplicates: bool = True):
    test = []
    for i in range(REF_NUM):
        with open(f'jfleg/test/test.ref{i}', 'r') as f:
            test.append(list(map(clean_str, f.readlines())))
    with open(f'jfleg/test/test.spellchecked.src', 'r') as f:
        test.append(list(map(clean_str, f.readlines())))

    test = np.array(test).flatten('F')
    test_values = [1. if i % (REF_NUM+1) != REF_NUM else 0. for i in range(len(test))]

    if change_student_values:
        for i in range(0, len(test), REF_NUM+1):
            if test[i+REF_NUM] in set(test[i:i+REF_NUM]):
                test_values[i+REF_NUM] = 1.

    not_none_idxs = [0] * len(test)
    if remove_duplicates:
        for i in range(0, len(test), REF_NUM+1):
            rev_list = list(np.flip(test[i:i+REF_NUM+1]))
            # last indexes of unique elements
            idx = [(len(rev_list) - rev_list.index(i) - 1) for i in set(test[i:i+REF_NUM+1])]

            assert REF_NUM in idx
            assert len(idx) > 1

            not_none_idxs[0] = idx[0] if idx[0] < REF_NUM else idx[1]
            for k in range(REF_NUM+1):
                if k not in idx:
                    test_values[i+k] = None

    jfleg = dict()
    count_all = 0
    for i in range(0, len(test), REF_NUM+1):
        count_all += add_item(
            jfleg, id='test', i=i, src=test[i+REF_NUM],
            refs=test[i:i+REF_NUM], not_none_id=not_none_idxs[i],
            values=test_values[i:i+REF_NUM+1])
    return jfleg


jfleg = create_dataset(
    change_student_values=True,
    remove_duplicates=True,
)
save_pickle(jfleg, 'data.pkl')
