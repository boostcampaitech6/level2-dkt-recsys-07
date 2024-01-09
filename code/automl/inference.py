from autogluon.tabular import TabularDataset, TabularPredictor
import os
import pandas as pd

# data 경로 설정
data_path = "../../data/"

# test Dataset 생성
test_df = pd.read_csv(os.path.join(data_path, "test_data.csv"))
test_df = test_df[test_df["answerCode"]==-1]
test_data = TabularDataset(test_df)

# load predictor

predictor = TabularPredictor.load("AutogluonModels/ag-20240109_091715")

# inference

answer = predictor.predict_proba(test_data)
test_df = pd.concat((test_df, answer), axis = 1)
# make a submission file
write_path = os.path.join("outputs/", "submission.csv")
os.makedirs(name="outputs", exist_ok=True)
with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(test_df.iloc[:,-1]):
            w.write("{},{}\n".format(id, p))