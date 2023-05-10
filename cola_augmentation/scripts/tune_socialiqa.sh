# This code QAC fine-tunes GPT2-Medium on SocialIQA dataset
# The model will be saved in data/tmp/gpt2-medium-no-trainer folder.

python3 gpt2_train.py --model_name_or_path gpt2-medium --train_file data/qa/socialiqa/train_qac.csv --output_dir data/tmp/gpt2-medium-no-trainer/ --per_device_train_batch_size 1 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100

# You also can use original version from CONDA authors:
# Usage: $ sh scripts/tune_socialiqa.sh 1 data/tmp 
# Uncomment the lines below and comment all lines above

# gpu=$1
# tmp=$2

# CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --model_name_or_path gpt2-medium --train_file data/qa/socialiqa/train_qac.csv --output_dir ${tmp}/gpt2-medium-no-trainer/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100