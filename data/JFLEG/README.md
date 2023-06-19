# Preparation of JfLEG Dataset

The main code can be found in `parse_data.py` file, the raw data was taken from `https://github.com/keisks/jfleg/test` folder. 

We marked sentences sentences from `test.src` (sentences written by students) as non-fluent and sentences from `test.ref*` (corrections of students' sentences) as fluent. The exception is case when the student's sentence matched one of the corrections. In this case we skipped this block of sentences. Additionally, we removed duplicate sentences. 

In `data.pkl` we stored same text as `ref`, `src` and `hypo` (see `data.pkl` structure in [readme](../README.md)) for each sample. In `data_with_student_src.pkl` we always stored sentence written by student as `src`, while `ref` and `hypo` contained same sentence, which could be annotator's sentence or student sentence itself.

## Usage

Download data [[link](https://github.com/keisks/jfleg)] and run `parse_data.py` script.

```Bash
git clone https://github.com/keisks/jfleg.git
python3 parse_data.py
rm -r jfleg
```

If you wish to recreate `data_with_student_src.pkl`, change `student_src` argument inside `parse_data.py` and rename the file name.