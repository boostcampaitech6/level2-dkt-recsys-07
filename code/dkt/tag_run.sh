#!/bin/bash


# model 지정
model = "lstm"

for var in {1..9};do
    train_name="train_data_tag"$var".csv"
    test_name="test_data_tag"$var".csv"
    
    # Train
    python train.py file_name=$train_name test_file_name=$test_name model_name="tag"$var.pt
    
    # Inference
    python inference.py model_name="tag"$var.pt
    
    # Output rename
    mv ./outputs/${model}_submission.csv ./outputs/${model}_submission_tag$var.csv
done

# out_dir 지정
out_dir = ./outputs/

python tag_ensemble.py +out_dir=$out_dir +out_name=${model}_submission