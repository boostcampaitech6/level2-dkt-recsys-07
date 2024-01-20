import numpy as np
import pandas as pd
import os
from datetime import datetime

dtype = {"userID": "int16", "answerCode": "int8", "KnowledgeTag": "int16"}
DATA_PATH = "../../data/"
DATA_FILE = "test_data.csv"
DATA = os.path.join(DATA_PATH, DATA_FILE)

df = pd.read_csv(DATA, dtype=dtype, parse_dates=["Timestamp"])
df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)

print("FE 시작:", datetime.now())

# 문제별 대분류 태그 추가
print("문제별 대분류 태그 추가")
df["BigTag"] = df["assessmentItemID"].str[2]

# 각 문제별 정답률 계산 후 추가
print("각 문제별 정답률 계산 후 추가")
answer_rates = df.groupby("assessmentItemID")["answerCode"].agg(["sum", "count"])
answer_rates["prob_correct_rate"] = answer_rates["sum"] / answer_rates["count"]
df = df.merge(answer_rates["prob_correct_rate"], on="assessmentItemID", how="left")

# 유저의 정답 문제 개수, 유저의 전체 문제개수, 유저의 정답률 추가
print("유저의 정답 문제 개수, 유저의 전체 문제개수, 유저의 정답률 추가")
df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(
    lambda x: x.cumsum().shift(1)
)
df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]
df = df.fillna(0)


# 문항 순서와 문항 순서별 정답률 추가
def percentile(s):
    return np.sum(s) / len(s)


print("문항 순서와 문항 순서별 정답률 추가")
df["prob_order"] = df["assessmentItemID"].apply(lambda x: x[7:]).astype(int)
pronum_group = df.groupby("prob_order")["answerCode"].agg([percentile])
pronum_group.columns = ["prob_order_correct_rate"]
df = pd.merge(df, pronum_group, on="prob_order", how="left")

# 문항 순서에 따른 난이도 추가
print("문항 순서에 따른 난이도 추가")
prob_difficulty = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 8,
    8: 11,
    9: 9,
    10: 7,
    11: 10,
    12: 12,
    13: 13,
}
df["prob_difficulty"] = df["prob_order"].apply(lambda x: prob_difficulty.get(x))

# 문제 풀이 세션 ver1/ver2(15min 제한) 동시에 계산
print("문제 풀이 세션 ver1/ver2(15min 제한) 동시에 계산")
temp_df = df.copy()
temp_df["time_diff"] = temp_df["Timestamp"].diff(-1).abs()
before_testId = "0"
before_time = ""
sess_ver1 = 0
sess_ver2 = 0
prob_list = []
temp_df["solving_session_ver1"] = 0
temp_df["solving_session_ver2"] = 0
for i, row in temp_df.iterrows():
    if before_testId != row["testId"] or row["prob_order"] in prob_list:
        before_testId = row["testId"]
        prob_list = []
        sess_ver1 += 1
        sess_ver2 += 1
    elif before_time > pd.Timedelta("15 minutes"):
        before_testId = row["testId"]
        prob_list = []
        sess_ver2 += 1
    temp_df.at[i, "solving_session_ver1"] = sess_ver1
    temp_df.at[i, "solving_session_ver2"] = sess_ver2
    before_time = row["time_diff"]
    prob_list.append(row["prob_order"])

# 소요시간 추가 ver1 은 범주형으로 ver2는 연속형으로 사용
print("소요시간 추가: ver1 은 범주형으로 ver2는 연속형으로 사용")
temp_df["time_diff_ver1"] = (
    temp_df.groupby("solving_session_ver1")["Timestamp"].diff(-1).abs()
)
df["time_diff_ver2"] = (
    temp_df.groupby("solving_session_ver2")["Timestamp"].diff(-1).abs()
)
df["time_diff_ver2"].fillna(pd.Timedelta(minutes=15), inplace=True)

# 문제 풀이 시작 시간
print("문제 풀이 시작 시간")


def calculate_time(dt):
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    return round(np.sin(np.pi * total_seconds / (24 * 3600)), 3)


df["solving_start_time"] = df["Timestamp"].apply(calculate_time)

# 문제 풀이 요일과 주말 여부
print("문제 풀이 요일과 주말 여부")
df["solving_day"] = df["Timestamp"].dt.weekday + 1
df["solving_is_weekend"] = df["solving_day"].apply(lambda x: 2 if x >= 6 else 1)

# 시험지(세션)별 총 시간 추가
print("시험지(세션)별 총 시간 추가")
temp_df["time_diff_session_ver2"] = df["time_diff_ver2"]
time_diff_group = temp_df.groupby("solving_session_ver2")[
    "time_diff_session_ver2"
].sum()
time_diff_group = time_diff_group.reset_index()
time_diff_group.columns = ["solving_session_ver2", "session_total_time"]
temp_df = temp_df.merge(time_diff_group, on="solving_session_ver2", how="left")
df["session_total_time"] = pd.to_timedelta(
    temp_df["session_total_time"]
).dt.total_seconds()

# 시험지 별 풀이시간/총시간
print("시험지 별 문제 (풀이 시간/총 소요시간) 비율 계산")
df["time_diff_ver2"] = pd.to_timedelta(df["time_diff_ver2"]).dt.total_seconds()
df["solving_time_rate"] = df["time_diff_ver2"] / df["session_total_time"]

# 소요시간 범주화, label encoding
print("소요시간 범주화, label encoding")
# nan: 6 으로 인코딩함
conditions = [
    temp_df["time_diff_ver1"] <= pd.Timedelta("5 seconds"),
    temp_df["time_diff_ver1"] <= pd.Timedelta("10 seconds"),
    temp_df["time_diff_ver1"] <= pd.Timedelta("1 minutes"),
    temp_df["time_diff_ver1"] <= pd.Timedelta("3 minutes"),
    temp_df["time_diff_ver1"] <= pd.Timedelta("5 minutes"),
    temp_df["time_diff_ver1"] > pd.Timedelta("5 minutes"),
]

choices = [1, 2, 3, 4, 5, 6]
df["time_diff_cate"] = np.select(conditions, choices, default=7)

# 유저마다 업데이트 되는 세션 추가
df["solving_session"] = temp_df.groupby("userID")["solving_session_ver2"].transform(
    lambda x: pd.factorize(x)[0] + 1
)

# 유저마다 업데이트 되는 이전 문제 정답 여부
df["prev_answer"] = df.groupby("userID")["answerCode"].shift(1) + 1
df["prev_answer"] = df["prev_answer"].fillna(0)

# 유저마다 업데이트 되는 태그 별 문제 풀이 개수 및 총 개수
print("유저마다 업데이트 되는 태그 별 문제 풀이 개수, 총 개수 및 정답 개수")
for tag in range(1, 10):
    temp_df[f"tag{tag}_count"] = temp_df.groupby("userID")["BigTag"].transform(
        lambda x: (x == str(tag)).cumsum()
    )
    temp_df[f"tag{tag}_total"] = temp_df.groupby("userID")["BigTag"].transform(
        lambda x: (x == str(tag)).sum()
    )
    temp_df[f"tag{tag}_correct"] = (
        temp_df.groupby("userID")
        .apply(lambda x: (x["BigTag"] == str(tag)) & (x["answerCode"] == 1))
        .cumsum()
        .reset_index(level=0, drop=True)
    )

# 하나의 컬럼에 BigTag 별 총 개수 넣기
df["total_tag_sum"] = temp_df.apply(
    lambda row: row[f'tag{row["BigTag"]}_total'], axis=1
)
df["total_tag_count"] = temp_df.apply(
    lambda row: row[f'tag{row["BigTag"]}_count'], axis=1
)
df["total_tag_correct"] = temp_df.apply(
    lambda row: row[f'tag{row["BigTag"]}_correct'], axis=1
)
df["tag_correct_rate"] = df.apply(
    lambda row: row["total_tag_correct"] / row["total_tag_sum"], axis=1
)
df["tag_correct_cum_rate"] = df.apply(
    lambda row: row["total_tag_correct"] / row["total_tag_count"], axis=1
)

print("FE 완료:", datetime.now())
print("df nan 개수")
print(df.isna().sum())
DATA = os.path.join(DATA_PATH, "FE_" + DATA_FILE)
df.to_csv(DATA, index=False)
