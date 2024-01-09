from autogluon.tabular import TabularDataset, TabularPredictor
import os
import pandas as pd

# data 경로 설정
data_path = "../../data/"

train_df = pd.read_csv(os.path.join(data_path, "train_data.csv"))
test_df = pd.read_csv(os.path.join(data_path, "test_data.csv"))
df = pd.concat((train_df, test_df))
train_df = df[df["answerCode"]>-1]
test_df = df[df["answerCode"]==-1]

# Read train data
train_data = TabularDataset(train_df)
test_data = TabularDataset(test_df)
# Train
label = "answerCode"
predictor = TabularPredictor(label=label, eval_metric="roc_auc", problem_type="binary").fit(train_data, gpu_nums = 1, time_limit=600, presets=["medium_quality"])
