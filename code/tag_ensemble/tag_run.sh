#!/bin/bash


# model 지정
#model="lstm"
model="lightgcn"

#model_dir="dkt"
model_dir="lightgcn"

# Train, Test data tag별로 생성 (dkt에서 실행)
if [ $model_dir == "dkt" ]; then
    python ./create_tag.py
fi

# config file 복사
mkdir config
## code 디렉토리에 model dir로 지정된 디렉토리의 모든 config파일을 복사합니다.
rm ./config/*.yaml
cp ../${model_dir}/config/*.yaml ./config

for var in {1..9};do
    train_name="train_data_tag"$var".csv"
    test_name="test_data_tag"$var".csv"
    
    # Train
    if [ $model_dir == "dkt" ]; then
        python ../${model_dir}/train.py file_name=$train_name test_file_name=$test_name model_name="tag"$var.pt
    elif [ $model_dir == "lightgcn" ]; then
        python ../${model_dir}/train.py model_filename="tag"$var.pt tag=$var
    fi

    # Inference
    if [ $model_dir == "dkt" ]; then
        python ../${model_dir}/inference.py model_name="tag"$var.pt
    elif [ $model_dir == "lightgcn" ]; then
        python ../${model_dir}/inference.py model_name="best_tag"$var.pt tag=$var
    fi

    
    
    # Output rename
    if [ $model_dir == "dkt" ]; then
        mv ./outputs/${model}_submission.csv ./outputs/${model}_submission_tag$var.csv
    elif [ $model_dir == "lightgcn" ]; then
        mv ./outputs/submission.csv ./outputs/${model}_submission_tag$var.csv
    fi
    
done

# out_dir 지정
out_dir=./outputs/

python ./tag_ensemble.py +out_dir=$out_dir +out_name=${model}_submission +test_file_name="test_data.csv"