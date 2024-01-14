import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from .valid import make_valid


class Preprocess:
    def __init__(self, args):
        self.args = args

        # 카테고리 사이즈 설정
        self.cate_sizes = {col: None for col in self.args.cate_cols}
        # 범주형 컬럼 설정
        self.cate_cols = self.args.cate_cols
        # 연속형 컬럼 설정
        self.cont_cols = self.args.cont_cols

        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self,
                   data: np.ndarray,
                   ratio: float = 0.7,
                   shuffle: bool = True,
                   seed: int = 0) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        if self.args.cv > 0:
            size = len(data)//5
            fold_index = self.args.cv
            if fold_index < 5:
                data_1 = np.concatenate((data[:size*(fold_index - 1)], data[size*fold_index:]))
                data_2 = data[size*(fold_index - 1):size*fold_index]
            elif fold_index == 5:
                data_1 = data[:size*(fold_index-1)]
                data_2 = data[size*(fold_index-1):]
            else:
                raise ValueError("args.cv not in [0,1,2,3,4,5]")
        else:
            size = int(len(data) * ratio)
            data_1 = data[:size]
            data_2 = data[size:]
        return data_1, data_2

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in self.cate_cols:
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)
                if col != "userID":
                    df[col] = df[col].apply(
                        lambda x: x if str(x) in le.classes_ else "unknown"
                    )
            if col != "userID":
                df[col] = df[col].astype(str)
                test = le.transform(df[col])
                df[col] = test

        def convert_time(s: str):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)
        return df

    def __feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Fill in if needed
        df['BigTag'] = df['assessmentItemID'].str[2]

        # 각 문제별 정답률 계산
        answer_rates = df.groupby('assessmentItemID')['answerCode'].agg(['sum', 'count'])
        answer_rates['rate'] = answer_rates['sum'] / answer_rates['count']

        # 원본 데이터프레임에 정답률 추가
        df = df.merge(answer_rates['rate'], on='assessmentItemID', how='left')
    
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
        
        df = df.fillna(0)

        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        self.args.cate_sizes = {}

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        # 각 범주형 변수의 고유 카테고리 개수 계산 및 저장
        for col in self.cate_cols:
            file_path = os.path.join(self.args.asset_dir, f"{col}_classes.npy")
            self.args.cate_sizes[col] = len(np.load(file_path))


        # 데이터 정렬
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)

        # 동적 컬럼 처리: 범주형 및 연속형 컬럼을 args에서 가져옴
        columns = self.cate_cols + self.cont_cols + ["answerCode"]
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: tuple(r[col].values for col in columns[1:])  # 첫 컬럼 'userID' 제외
            )
        )
        return group.values

    def load_train_data(self, file_name: str) -> None:
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name: str) -> None:
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, args):
        self.data = data
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.cate_cols = args.cate_cols  # 범주형 컬럼
        self.cont_cols = args.cont_cols  # 연속형 컬럼

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]

        # 범주형 및 연속형 데이터 분리
        x_cate_cols = self.cate_cols.copy()
        if "userID" in x_cate_cols:  # self.cate_cols의 복사본 생성
            x_cate_cols.remove("userID")  # 복사본에서 "userID" 제거
        total_cols = []
        total_cols = x_cate_cols +self.cont_cols
        
        data = {}

        for i, col in enumerate(total_cols):
            if col in x_cate_cols:
                data[col] = torch.tensor(row[i]+1, dtype=torch.int)
            else:
                data[col] = torch.tensor(row[i], dtype=torch.float)
                
        correct = row[-1]
        data["correct"] = torch.tensor(correct, dtype=torch.int)
        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > self.max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-self.max_seq_len:]
            mask = torch.ones(self.max_seq_len, dtype=torch.int16)
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros(self.max_seq_len)
                tmp[self.max_seq_len-seq_len:] = data[k]
                data[k] = tmp
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        data["mask"] = mask
        
        # Generate interaction
        interaction = data["correct"] + 1  # 패딩을 위해 correct값에 1을 더해준다.
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["interaction"] = interaction
        return data

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(args, train: np.ndarray, valid: np.ndarray) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader

