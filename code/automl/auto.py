from autogluon.tabular import TabularDataset, TabularPredictor
import os

# data 경로 설정
data_path = "../../data/"

# Read train data
train_data = TabularDataset(os.path.join(data_path, "train_data.csv"))
test_data = TabularDataset(os.path.join(data_path, "test_data.csv"))
# Train
label = "answerCode"
predictor = TabularPredictor(label=label, eval_metric="roc_auc", problem_type="binary").fit(train_data, presets=["high_quality"])

predictor.evaluate(test_data, silent=True) # 학습한 모든 모델의 예측값을 score 기준으로 가중치를 둔 최종 결과 확인
predictor.leaderboard(test_data) # 각 모델별로 score와 예측 시간 및 학습 시간 등 여러 값 확인

predictor.predict_proba(test_data)