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
(Categorical feature: *)
<!-- * userID 
* assessmentItemID           0
* testId                     0
* answerCode                 0
* Timestamp                  0
* KnowledgeTag               0 -->
* *BigTag                     : 문제 별 대분류 태그 (1~9)
* prob_correct_rate          : 문제 별 정답률
* user_correct_answer        : 유저 별 정답 문제 개수
* user_total_answer          : 유저 별 총 문제 개수
* user_acc                   : 유저 별 정답률
* prob_order                 : 문항 순서(1~13)
* prob_order_correct_rate    : 문항 순서 별 정답률
* *prob_difficulty           : 문항 순서 별 난이도(1~13) 
* time_diff_ver2             : 문제 별 풀이 소요 시간
* solving_start_time         : 문제 풀이 시작 시간(sin 함수 사용)
* solving_start_time_r3      : 문제 풀이 시작 시간(sin 함수 사용, 3자리 반올림)
* *solving_day               : 문제 풀이 요일(1~7)
* *solving_is_weekend        : 문제 풀이 주말 여부(평일 1, 주말 2)
* session_total_time         : 시험지(풀이 세션) 별 총 소요시간
* solving_time_rate          : (time_diff_ver2/session_total_time) 한 풀이 소요 시간 비율
* *time_diff_cate            : time_diff_ver2의 범주형 변수(1~7/5초,10초,1분,3분,5분 기준)
* solving_session            : 사용자 마다 업데이트 되는 시험지(풀이 세션) 번호(개수)
* *prev_answer               : 사용자의 이전 문제 정답 여부 (첫번째 값 0, 오답 1, 정답 2)
* total_tag_sum              : 사용자의 태그별 문제풀이 총 개수
* total_tag_count            : 사용자의 태그별 문제풀이 누적 개수
* total_tag_correct          : 사용자의 태그별 정답 누적 개수
* tag_correct_rate           : 사용자의 태그별 정답률(total_tag_correct/total_tag_sum)
* tag_correct_cum_rate       : 사용자의 태그별 정답률(total_tag_correct/total_tag_count)


