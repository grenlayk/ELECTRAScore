# This code is for Domain Adaptation on CoLA dataset
# GPT2-Medium fine-tuned on SocialIQA dataset is available here:
# https://huggingface.co/grenlayk/gpt2-medium-socialiqa
#
# The output df_gen_topk_{num_tries}_sampling.csv will be stored in data/tmp
# Usage: $ sh scripts/tune_cola.sh 1000 

num_tries=$1

python3 gpt2_train.py --seed 13 --model_name_or_path grenlayk/gpt2-medium-socialiqa --train_file data/cls/cola/train_qac_13.csv --output_dir data/tmp/gpt2-medium-socialiqa-8-seed-13/ --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100

python3 generate_dataset.py --dataset_name cola --seed 13 --model_name_or_path data/tmp/gpt2-medium-socialiqa-8-seed-13/ --output_dir data/tmp --strategy topk --num_tries ${num_tries}

python3 parse.py data/tmp/df_gen_topk_${num_tries}_sampling.csv

# You also can use original version from CONDA authors:
# Usage: $ sh scripts/tune_cola.sh 1 data/tmp cola
# Uncomment the lines below and comment all lines above

# gpu=$1
# tmp=$2
# dataset=$3


# CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 13 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/cls/${dataset}/train_qac_13.csv --output_dir ${tmp}/gpt2-medium-socialiqa-8-seed-13/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100

# CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset.py --dataset_name ${dataset} --seed 13 --model_name_or_path ${tmp}/gpt2-medium-socialiqa-8-seed-13/ --output_dir ${tmp} --strategy topk --num_tries 450
