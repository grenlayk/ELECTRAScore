# CoLA extension

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

If you want to create your own data slice for a domain adaptation, refer to `generate_cola_samples.ipynb` notebook.

To tune GPT2-Medium on your own QA dataset refer to `scripts/tune_socialiqa.sh`. QAC fine-tuned GPT2-Medium on SocialIQA dataset is available at [HuggingFace](https://huggingface.co/grenlayk/gpt2-medium-socialiqa).

For domain adaptation on CoLA dataset and data generation run this script:
```shell script
$ scripts/tune_cola.sh 1500
```
where first argument is number of generated samples.

## CoLA extension

We conduct post-processing for data generated with `tune_cola.sh` script. We use ChatGPT as a markup supervisor.

The (chatgpt_scoring.ipynb)[chatgpt_scoring.ipynb] notebook contains following steps:
- Testing of prompt on original CoLA dataset
- Scoring each row of generated data with ChatGPT through OpenAI API.
- Creation of CoLA-E and CoLA-ECL datasets. 
    - CoLA-E (CoLA-Extended) contains only rows for which generated label matched ChatGPT label.
    - CoLA-ECL (CoLA-Extended with ChatGPT Labels) contains all rows with ChatGPT labels.
- Creation of balanced versions of CoLA-E and CoLA-ECL datasets and supporting visualizations.


### Data

Here we provide links to files which we used and created:
- Raw generated samples ([link](https://drive.google.com/file/d/1mTobfflscj6xKVGmlmEf1qJAMUNSsXbT/view?usp=share_link))
- Parsed generated samples ([link](https://drive.google.com/file/d/1iL5aYEk01vsv6GLbV_5Cs97Lzys1yhYb/view?usp=sharing))
- Parsed generated samples with ChatGPT labels ([link](https://drive.google.com/file/d/1txYNeceA8FMHqeqFiDAI4WwGfJ1dadhe/view?usp=share_link))
- Generated part of CoLA-E dataset ([link](https://drive.google.com/file/d/1zjlzKbHU7pkuEaVqGkES5DosB52k6Upk/view?usp=share_link))
- Generated part of CoLA-ECL dataset ([link](https://drive.google.com/file/d/1edCFZQcqoAIQEHhdDZqdZUt5Id4itbX1/view?usp=share_link))
- Balanced CoLA-E dataset ([link](https://drive.google.com/file/d/1xwSOFmcK3HLjqEmNezMSZrcGzbADBsdh/view?usp=share_link))
- Balanced CoLA-ECL dataset ([link](https://drive.google.com/file/d/1qCcsSnwZ9ef7S8vRVXKF6MMj9_LpN7T1/view?usp=share_link))
