#!/bin/bash

mkdir data
mkdir data/tmp

# upload CoLA data (in_domain_train.tsv)
wget https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/raw/in_domain_train.tsv
# upload generated data (generated_data.csv)
gdown 1iL5aYEk01vsv6GLbV_5Cs97Lzys1yhYb

mv in_domain_train.tsv data/tmp
mv generated_data.csv data/tmp

# optional step
gdown 1txYNeceA8FMHqeqFiDAI4WwGfJ1dadhe
mv labeled_generated_data.csv data/tmp

wandb login
