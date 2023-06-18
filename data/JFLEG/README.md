# Preparation of JfLEG Dataset

The main code can be found in `parse_data.py` file, the raw data was taken from `https://github.com/keisks/jfleg/test` folder. 

We marked sentences sentences from `test.src` (sentences written by students) as non-fluent and sentences from `test.ref*` (corrections of students' sentences) as fluent. The exception is case when the student's sentence matched one of the corrections. In this case we skipped this block of sentences. Additionally, we removed duplicate sentences. 

We stored same text as `ref`, `src` and `hypo` (see `data.pkl` structure in [readme](../README.md)) for each sample.

## Usage

Download data [[link](https://github.com/keisks/jfleg)] and run `parse_data.py` script.

```Bash
git clone https://github.com/keisks/jfleg.git
python3 parse_data.py
rm -r jfleg
```