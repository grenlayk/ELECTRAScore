# CoLA extention

This folder contains code to generate extra data for [CoLA dataset](https://nyu-mll.github.io/CoLA/). 

This code is a modified version of [CONDA repository](https://github.com/dheeraj7596/CONDA), the [Leveraging QA Datasets to Improve Generative Data Augmentation](https://arxiv.org/pdf/2205.12604.pdf) paper by Mekala, Dheeraj and Vu, Tu and Schick, Timo and Shang, Jingbo.

## Training

### Required Inputs
The framework requires QA datasets and target-task classification datasets.
* All datasets are in `csv` format.
* QA datasets are in `data/qa/` folder. Each sample is in Question-Answer-Context format.
* Target classification datasets are in `data/cls/` folder.
  * Each classification dataset has `train` sub-folder.
  * `Train` folder have files corresponding to few-shot supervision.
  * Note that `train_qac_x.csv` is same as `train/train_x.csv` in QAC format, that is used for domain adaptation step. 

### Commands

If you want to create your own data slice for a domain adoptation, refer to `generate_cola_samples.ipynb` notebook.

To tune GPT2-Medium on your own QA dataset refer to `scripts/tune_socialiqa.sh`. QAC fine-tuned GPT2-Medium on SocialIQA dataset is available at [HuggingFace](https://huggingface.co/grenlayk/gpt2-medium-socialiqa).

For domain adaptation on CoLA dataset and data generarion run this script:
```shell script
$ scripts/tune_cola.sh 1500
```
where first argument is number of generated samples.
