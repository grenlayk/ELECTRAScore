import pandas as pd
import pickle
import re


def prepare_data(path):
    data = pd.read_csv(path, sep='\t')
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    data['target'] = data['target'].apply(lambda row: tag_re.sub('', row))
    data_negative = data[data.category.str.contains("Fluency")]
    data_positive = data[data.category == 'No-error']
    data_negative['label'] = 0.
    data_positive['label'] = 1.
    return pd.concat([data_negative, data_positive])


def add_items(data_dict, doc_id, idx, sys_name, src_text, ref_text, text, value):
    data_dict[f'{doc_id}_{idx}'] = {
        'src': None,
        'ref': None,
        "better": {},
    }
    data_dict[f'{doc_id}_{idx}']['src'] = src_text
    data_dict[f'{doc_id}_{idx}']['ref'] = ref_text
    data_dict[f'{doc_id}_{idx}']['better'] = {
        "sys_name": sys_name,
        "sys": text,
        "scores": {'fluency': value}
    }


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def save_to_pkl(data_df, file_name):
    df_grouped = data_df.groupby('source')
    data_dict = dict()

    for idx, (group_name, df_group) in enumerate(df_grouped):
        for row_index, row in df_group.iterrows():
            add_items(
                data_dict=data_dict,
                doc_id=idx,
                idx=row_index,
                sys_name=row['system'],
                src_text=group_name,
                ref_text=group_name,
                text=row['target'],
                value=row['label']
            )

    save_pickle(data_dict, file_name)


data_2020 = prepare_data('data/mqm_newstest2020_zhen.tsv')
data_2021 = prepare_data('data/mqm_newstest2021_zhen.tsv')
data_ted = prepare_data('data/mqm_ted_zhen.tsv')

save_to_pkl(data_2020, 'data_2020.pkl')
save_to_pkl(data_2021, 'data_2021.pkl')
save_to_pkl(data_ted, 'data_ted.pkl')
