#!/bin/bash


# model 지정
model="lstm"
model_dir="dkt"

# Train, Test data tag별로 생성
python ./create_tag.py

# config file 복사
mkdir config
## code 디렉토리에 model dir로 지정된 디렉토리의 모든 config파일을 복사합니다.
cp ../${model_dir}/config/*.yaml ./config

for var in {1..9};do
    train_name="train_data_tag"$var".csv"
    test_name="test_data_tag"$var".csv"
    
    # Train
    python ../${model_dir}/train.py file_name=$train_name test_file_name=$test_name model_name="tag"$var.pt
    
    # Inference
    python ../${model_dir}/inference.py model_name="tag"$var.pt
    
    # Output rename
    mv ./outputs/${model}_submission.csv ./outputs/${model}_submission_tag$var.csv
done

# out_dir 지정
out_dir=./outputs/

python ./tag_ensemble.py +out_dir=$out_dir +out_name=${model}_submission