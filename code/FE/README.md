# Feature Engineering

## Setup
```bash
cd /opt/ml/input/code/FE
(dkt) python feature_engineering.py
```

## Files
`code/FE`
* `feature_engineering.py`: 학습코드입니다.

## Option
FE 대상 데이터(train/test) 파일 변경 시 코드 수정 필요함

## Feature Description
<!-- * userID 
* assessmentItemID           0
* testId                     0
* answerCode                 0
* Timestamp                  0
* KnowledgeTag               0 -->
* BigTag                     : 문제 별 대분류 태그 (1~9)
* prob_correct_rate          : 문제 별 정답률
* user_correct_answer        : 유저 별 정답 문제 개수
* user_total_answer          : 유저 별 총 문제 개수
* user_acc                   : 유저 별 정답률
* prob_order                 : 문항 순서
* prob_order_correct_rate    : 문항 순서 별 정답률
* prob_difficulty            : 문항 순서 별 난이도(1~13) 
* time_diff_ver2             : 문제 별 풀이 소요 시간
* session_total_time         : 시험지(풀이 세션) 별 총 소요시간
* rest_time                  : (session_total_time - time_diff_ver2) 한 잔여 시간
* time_diff_cate             : time_diff_ver2의 범주형 변수(0~6/5초,10초,1분,3분,5분 기준)