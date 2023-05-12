# %%
import pickle

import nltk
from mosestokenizer import MosesDetokenizer

nltk.download('stopwords')
nltk.download('punkt')
detokenizer = MosesDetokenizer('en')


def read_file_to_list(file_name):
    lines = []
    with open(file_name, 'r', encoding='utf8') as f:
        for line in f.readlines():
            lines.append(line.strip())
    return lines


def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved to {file}.')


def detokenize(text: str):
    words = text.split(" ")
    return detokenizer(words)
