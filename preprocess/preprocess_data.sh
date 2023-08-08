#!/usr/bin/env bash

export PROJECT_ROOT=$project_dir
export DATA_PATH=$data_folder

# process raw data and write it into .txt file
python ace/read_args_with_entity_ace.py
python ere/read_args_with_entity_ere.py

# save data into TensorDataset for prompt type
python ace/save_dataset_sl.py --out_dir $output_dir --data_dir $data_dir --prompts $prompt_type
python ace/save_dataset_few_zero.py --few_shot True --out_dir $output_dir --data_dir $data_dir --prompts $prompt_type
python ace/save_dataset_few_zero.py --zero_shot True --out_dir $output_dir --data_dir $data_dir --prompts $prompt_type

python ere/save_dataset_sl.py --out_dir $output_dir --data_dir $data_dir --prompts $prompt_type
python ere/save_dataset_few_zero.py --few_shot True --out_dir $output_dir --data_dir $data_dir --prompts $prompt_type
python ere/save_dataset_few_zero.py --zero_shot True --out_dir $output_dir --data_dir $data_dir --prompts $prompt_type
