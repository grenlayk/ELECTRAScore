# Preparation of CONLL Dataset

The main code can be found in `parse_data.py` file, the raw data was taken from `conll14st-test-data/alt/official-2014.combined-withalt.m2` file. 

We marked sentences without any annotations as fluent, otherwise as not fluent. The exception is case when the sentence was marked as correct and incorrect at the same time by different annotators. In this case, we skipped these sentences, without adding them to the dataset.

We stored same text as `ref`, `src` and `hypo` (see `data.pkl` structure in [readme](../README.md)) for each sample.

## Usage

Download data [[link](https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz)] and run `parse_data.py` script.

```Bash
wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz
tar –xvzf conll14st-test-data.tar.gz
python3 parse_data.py
```