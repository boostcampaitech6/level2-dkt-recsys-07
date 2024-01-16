import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
} 
DATA_PATH = '../../data/'
DATA = os.path.join(DATA_PATH, "train_data.csv")

df = pd.read_csv(DATA, dtype=dtype, parse_dates=['Timestamp'])
df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)

print('FE 시작:',datetime.now())

# 문제별 대분류 태그 추가
print('문제별 대분류 태그 추가')
df['BigTag'] = df['assessmentItemID'].str[2]

# 각 문제별 정답률 계산 후 추가
print('각 문제별 정답률 계산 후 추가')
answer_rates = df.groupby('assessmentItemID')['answerCode'].agg(['sum', 'count'])
answer_rates['prob_correct_rate'] = answer_rates['sum'] / answer_rates['count']
df = df.merge(answer_rates['prob_correct_rate'], on='assessmentItemID', how='left')

# 유저의 정답 문제 개수, 유저의 전체 문제개수, 유저의 정답률 추가
print('유저의 정답 문제 개수, 유저의 전체 문제개수, 유저의 정답률 추가')
df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
df = df.fillna(0)

# 문항 순서와 문항 순서별 정답률 추가
def percentile(s):
    return np.sum(s) / len(s)

print('문항 순서와 문항 순서별 정답률 추가')
df['prob_order'] = df['assessmentItemID'].apply(lambda x: x[7:]).astype(int)
pronum_group=df.groupby('prob_order')['answerCode'].agg([percentile])
pronum_group.columns = ['prob_order_correct_rate']
df = pd.merge(df, pronum_group, on='prob_order',how="left")

# 문항 순서에 따른 난이도 추가
print('문항 순서에 따른 난이도 추가')
prob_difficulty={1:1,2:2,3:3,4:4,5:5,6:6,7:8,8:11,9:9,10:7,11:10,12:12,13:13}
df['prob_difficulty']= df['prob_order'].apply(lambda x: prob_difficulty.get(x))

# 문제 풀이 세션 ver1/ver2(15min 제한) 동시에 계산
print('문제 풀이 세션 ver1/ver2(15min 제한) 동시에 계산')
temp_df = df.copy()
temp_df['time_diff']= temp_df['Timestamp'].diff(-1).abs()
before_testId = '0'
before_time = ''
sess_ver1=0
sess_ver2=0
prob_list=[]
temp_df['solving_session_ver1']=0
temp_df['solving_session_ver2']=0
for i,row in temp_df.iterrows():
    if  before_testId != row['testId'] or row['prob_order'] in prob_list:
        before_testId=row['testId']
        prob_list=[]
        sess_ver1+=1
        sess_ver2+=1
    elif before_time > pd.Timedelta('15 minutes'):
        before_testId=row['testId']
        prob_list=[]
        sess_ver2+=1
    temp_df.at[i, 'solving_session_ver1'] = sess_ver1
    temp_df.at[i, 'solving_session_ver2'] = sess_ver2
    before_time = row['time_diff']
    prob_list.append(row['prob_order'])

# 소요시간 추가 ver1 은 범주형으로 ver2는 연속형으로 사용
print('소요시간 추가: ver1 은 범주형으로 ver2는 연속형으로 사용')
temp_df['time_diff_ver1']= temp_df.groupby('solving_session_ver1')['Timestamp'].diff(-1).abs()
df['time_diff_ver2']= temp_df.groupby('solving_session_ver2')['Timestamp'].diff(-1).abs()
df['time_diff_ver2'].fillna(pd.Timedelta(minutes=15),inplace=True)

# 시험지(세션)별 총 시간 추가
print('시험지(세션)별 총 시간 추가')
temp_df['time_diff_session_ver2']=df['time_diff_ver2']
time_diff_group=temp_df.groupby('solving_session_ver2')['time_diff_session_ver2'].sum()
time_diff_group = time_diff_group.reset_index()
time_diff_group.columns = ['solving_session_ver2', 'session_total_time']
temp_df = temp_df.merge(time_diff_group, on='solving_session_ver2',how='left')
df['session_total_time'] =  pd.to_timedelta(temp_df['session_total_time']).dt.total_seconds()

# 시험지 별 나머지 시간 계산
print('시험지 별 나머지 시간 계산')
df['time_diff_ver2']= pd.to_timedelta(df['time_diff_ver2']).dt.total_seconds()
df['rest_time']=df['session_total_time']-df['time_diff_ver2']

# 연속형 시간 변수 scale
print('연속형 시간 변수 scale')
scaler = MinMaxScaler()
df['time_diff_ver2'] =scaler.fit_transform(df['time_diff_ver2'].values.reshape(-1, 1))
df['rest_time'] = scaler.fit_transform(df['rest_time'].values.reshape(-1, 1))
df['session_total_time'] = scaler.fit_transform(df['session_total_time'].values.reshape(-1, 1))



print('소요시간 범주화, label encoding')
conditions = [
    temp_df['time_diff_ver1'] <= pd.Timedelta('5 seconds'),
    temp_df['time_diff_ver1'] <= pd.Timedelta('10 seconds'),
    temp_df['time_diff_ver1'] <= pd.Timedelta('1 minutes'),
    temp_df['time_diff_ver1'] <= pd.Timedelta('3 minutes'),
    temp_df['time_diff_ver1'] <= pd.Timedelta('5 minutes'),
    temp_df['time_diff_ver1'] > pd.Timedelta('5 minutes')
]

choices = [0, 1, 2, 3, 4, 5]
df['time_diff_cate'] = np.select(conditions, choices, default=6)

print('FE 완료:',datetime.now())
print('df nan 개수')
print(df.isna().sum())
DATA = os.path.join(DATA_PATH, "FE_train_data.csv")
df.to_csv(DATA,index=False)