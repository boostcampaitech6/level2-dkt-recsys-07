import numpy as np
import pandas as pd
import os
from datetime import datetime

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


# 문항 순서에 따른 난이도 추가
print('문항 순서에 따른 난이도 추가')
prob_difficulty={'001':1,'002':2,'003':3,'004':4,'005':5,'006':6,'007':8,'008':11,'009':9,'010':7,'011':10,'012':12,'013':13}
df['prob_difficulty']=df['assessmentItemID'].apply(lambda x: prob_difficulty.get(x[-3:]))


# 문제 풀이 세션 ver1/ver2(15min 제한) 동시에 계산
print('문제 풀이 세션 ver1/ver2(15min 제한) 동시에 계산')
temp_df = df.copy()
temp_df['order_of_prob']=temp_df['assessmentItemID'].apply(lambda x: x[7:]).astype(int)
temp_df['time_diff']= temp_df['Timestamp'].diff(-1).abs()
before_testId = '0'
before_time = ''
sess_ver1=0
sess_ver2=0
prob_list=[]
temp_df['solving_session_ver1']=0
temp_df['solving_session_ver2']=0
for i,row in temp_df.iterrows():
    if  before_testId != row['testId'] or row['order_of_prob'] in prob_list:
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
    prob_list.append(row['order_of_prob'])

# 소요시간 추가 ver1 은 범주형으로 ver2는 연속형으로 사용
print('소요시간 추가: ver1 은 범주형으로 ver2는 연속형으로 사용')
temp_df['time_diff_ver1']= temp_df.groupby('solving_session_ver1')['Timestamp'].diff(-1).abs()
df['time_diff_ver2']= temp_df.groupby('solving_session_ver2')['Timestamp'].diff(-1).abs()
df['time_diff_ver2'].fillna(pd.Timedelta(minutes=15),inplace=True)

print('소요시간 범주화')
df['5_sec']=0
df['10_sec']=0
df['1_min']=0
df['3_min']=0
df['5_min']=0
df['more_5_min']=0
df['nan']=0
for i,row in temp_df.iterrows():
    if row['time_diff']<=pd.Timedelta('5 seconds'):
        df.at[i,'5_sec']=1
    elif row['time_diff']<=pd.Timedelta('10 seconds'):
        df.at[i,'10_sec']=1
    elif row['time_diff']<=pd.Timedelta('1 minutes'):
        df.at[i,'1_min']=1
    elif row['time_diff']<=pd.Timedelta('3 minutes'):
        df.at[i,'3_min']=1
    elif row['time_diff']<=pd.Timedelta('5 minutes'):
        df.at[i,'5_min']=1
    elif row['time_diff'] > pd.Timedelta('5 minutes'):
        df.at[i,'more_5_min']=1
    else:
        df.at[i,'nan']=1

print('FE 완료:',datetime.now())
print('df nan 개수')
print(df.isna().sum())
DATA = os.path.join(DATA_PATH, "FE_train_data.csv")
df.to_csv(DATA,index=False)