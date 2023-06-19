import pickle
import numpy as np


REF_NUM = 4


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def clean_str(text: str):
    return text.strip().replace('\n', ' ')


def add_item(jfleg, block_id, idx, src_text, text, value):
    jfleg[f'{block_id}_{idx}'] = {
        'src': None,
        'ref_summ': None,
        'sys_summs': {
             f'JFLEG': {
                 'sys_summ': None,
                 'scores': {'fluency': None}
             }
         }
    }
    jfleg[f'{block_id}_{idx}']['src'] = src_text
    jfleg[f'{block_id}_{idx}']['ref_summ'] = text
    jfleg[f'{block_id}_{idx}']['sys_summs'][f'JFLEG']['sys_summ'] = text
    jfleg[f'{block_id}_{idx}']['sys_summs'][f'JFLEG']['scores']['fluency'] = value


def create_dataset(
        remove_correct_students: bool = True,
        remove_duplicates: bool = True,
        student_src: bool = False):
    test = []
    for i in range(REF_NUM):
        with open(f'jfleg/test/test.ref{i}', 'r') as f:
            test.append(list(map(clean_str, f.readlines())))
    with open(f'jfleg/test/test.src', 'r') as f:
        test.append(list(map(clean_str, f.readlines())))

    test = np.array(test).flatten('F')
    test_values = [1. if i % (REF_NUM+1) != REF_NUM else 0. for i in range(len(test))]

    texts = []
    values = []
    for i in range(0, len(test), REF_NUM+1):
        if not (remove_correct_students and test[i+REF_NUM] in set(test[i:i+REF_NUM])):
            texts.extend(test[i:i+REF_NUM+1])
            values.extend(test_values[i:i+REF_NUM+1])

    grouped_texts = []
    grouped_values = []
    for i in range(0, len(texts), REF_NUM+1):
        text_group = [texts[i+REF_NUM]]
        value_group = [0.]
        added_texts = texts[i:i+REF_NUM]
        if remove_duplicates:
            added_texts = np.unique(texts[i:i+REF_NUM])
        text_group.extend(added_texts)
        value_group.extend([1.] * len(added_texts))
        grouped_texts.append(text_group)
        grouped_values.append(value_group)

    jfleg = dict()
    for i, (text_group, value_group) in enumerate(zip(grouped_texts, grouped_values)):
        for j, (text, value) in enumerate(zip(text_group, value_group)):
            src_text = text
            if student_src:
                src_text = text_group[0]
            add_item(jfleg, i, j, src_text, text, value)
    return jfleg


jfleg = create_dataset(
    remove_correct_students=True,
    remove_duplicates=True,
    student_src=False,
)
save_pickle(jfleg, 'data.pkl')
