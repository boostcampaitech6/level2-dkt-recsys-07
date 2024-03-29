import os
import random
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from .valid import make_valid

from torch.utils.data import TensorDataset
from tqdm import tqdm


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(
        self, data: np.ndarray, ratio: float = 0.7, shuffle: bool = True, seed: int = 0
    ) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        if self.args.cv > 0:
            size = len(data) // 5
            fold_index = self.args.cv
            if fold_index < 5:
                data_1 = np.concatenate(
                    (data[: size * (fold_index - 1)], data[size * fold_index :])
                )
                data_2 = data[size * (fold_index - 1) : size * fold_index]
            elif fold_index == 5:
                data_1 = data[: size * (fold_index - 1)]
                data_2 = data[size * (fold_index - 1) :]
            else:
                raise ValueError("args.cv not in [0,1,2,3,4,5]")
        else:
            size = int(len(data) * ratio)
            data_1 = data[:size]
            data_2 = data[size:]
        return data_1, data_2

    def split_data_sequantially(
        self, data: np.ndarray, ratio: float = 0.7, seed: int = 0
    ) -> Tuple[np.ndarray]:
        """
        split data into two parts with a given ratio.
        """
        last_sequence_length = 0
        train_user_data = []
        valid_user_data = []
        user_rows = []
        for i, rows in enumerate(data):
            if len(rows[0]) < last_sequence_length:
                size = int(len(user_rows) * ratio)
                train_user_data.extend(user_rows[:size])
                valid_user_data.extend(user_rows[size:])
                user_rows = []
            user_rows.append(rows)
            last_sequence_length = len(rows[0])
        train_data = np.empty(len(train_user_data), dtype=object)
        valid_data = np.empty(len(valid_user_data), dtype=object)
        for i, rows in enumerate(train_user_data):
            train_data[i] = rows
        for i, rows in enumerate(valid_user_data):
            valid_data[i] = rows
        return train_data, valid_data

    def __save_labels(self, encoder: LabelEncoder, name: str) -> None:
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        df["Raw_userID"] = df["userID"]

        # 연속형 변수 minmaxscaler
        scaler = MinMaxScaler()
        for col in self.args.cont_cols:
            if col == "Timestamp":
                continue
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

        if self.args.model.name[:3] == "GCN":
            path1 = os.path.join(self.args.data_dir, self.args.file_name)
            path2 = os.path.join(self.args.data_dir, self.args.test_file_name)

            data1 = pd.read_csv(path1)
            data2 = pd.read_csv(path2)

            data = pd.concat([data1, data2])
            data.drop_duplicates(
                subset=["userID", "assessmentItemID"], keep="last", inplace=True
            )

            userid, itemid = (
                sorted(list(set(data.userID))),
                sorted(list(set(data.assessmentItemID))),
            )
            n_user, n_item = len(userid), len(itemid)

            userid2index = {v: i + 1 for i, v in enumerate(userid)}  # 0 은 패딩이므로 비워둡니다.
            itemid2index = {v: i + n_user + 1 for i, v in enumerate(itemid)}
            id2idx = dict(userid2index, **itemid2index)
            id2idx["unknown"] = 0

            for col in ["userID", "assessmentItemID"]:
                df[col] = df[col].map(id2idx)

        for col in self.args.cate_cols:
            if (
                col in ["userID", "assessmentItemID"]
                and self.args.model.name[:3] == "GCN"
            ):
                continue
            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
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
        return df

    def load_data_from_file(self, file_name: str, is_train: bool = True) -> np.ndarray:
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        self.args.n_cate = {
            col: len(np.load(os.path.join(self.args.asset_dir, col + "_classes.npy")))
            for col in self.args.cate_cols
        }

        # MF는 group을 만들지 않고 return
        if self.args.model.name.lower() in ["mf", "lmf"]:
            df = df[["userID", "assessmentItemID", "answerCode"]]
            if not is_train:
                df = df[df["answerCode"] == -1]
            return df.values

        df = df.sort_values(by=["Raw_userID", "Timestamp"], axis=0)
        self.args.columns = (
            ["Raw_userID", "answerCode"] + self.args.cate_cols + self.args.cont_cols
        )
        group = (
            df[self.args.columns]
            .groupby("Raw_userID")
            .apply(lambda r: (tuple(r[col].values for col in self.args.columns[1:])))
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
        self.max_seq_len = args.model.max_seq_len

    def __getitem__(self, index: int) -> dict:
        row = self.data[index]
        return row

    def __len__(self) -> int:
        return len(self.data)


def get_loaders(
    args, train: np.ndarray, valid: np.ndarray
) -> Tuple[torch.utils.data.DataLoader]:
    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        if args.model.name.lower() in ["mf", "lmf"]:
            trainset = TensorDataset(torch.LongTensor(train))
        else:
            trainset = DKTDataset(prepare(train, args), args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )
    if valid is not None:
        if args.model.name.lower() in ["mf", "lmf"]:
            valset = TensorDataset(torch.LongTensor(valid))
        else:
            valset = DKTDataset(prepare(valid, args), args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
        )

    return train_loader, valid_loader


def sliding_window(args, data: np.ndarray) -> np.ndarray:
    if args.session_aug:
        old_len = len(data)
        session_idx = args.columns[1:].index("solving_session")
        stack = []
        for user in tqdm(data):
            session = user[session_idx][0]
            for idx, ss in enumerate(user[session_idx]):
                if ss != session:
                    session = ss
                    stack.append(tuple([col[:idx] for col in user]))
            stack.append(user)
        data = np.empty(len(stack), dtype=object)
        for i, row in enumerate(stack):
            data[-(i + 1)] = row
        print(f"Augmentated from {old_len} to {len(stack)}")
    elif args.random_aug:
        stack = []
        for user in tqdm(data):
            l = len(user[0])
            if l < args.max_seq_len:
                stack.append(user)
                continue

            for _ in range((l // args.max_seq_len) ** 2):
                ind = np.random.choice(l, args.max_seq_len, replace=False)
                ind.sort()
                stack.append(tuple([r[ind] for r in user]))
        data = np.empty(len(stack), dtype=object)
        for i, row in enumerate(stack):
            data[-(i + 1)] = row
    elif args.stride > 0:
        old_len = len(data)
        stack = []
        for user in tqdm(data):
            stack.append(user)
            last = args.stride
            while len(user[0]) - last > 14:
                stack.append(tuple([r[:-last] for r in user]))
                last += args.stride
        data = np.empty(len(stack), dtype=object)
        for i, rows in enumerate(stack):
            data[-(i + 1)] = rows
        print(f"Augmentated from {old_len} to {len(stack)}")
    return data


def prepare(dataset, args):
    max_seq_len = args.model.max_seq_len
    print("prepare data for dataset")
    for i, row in tqdm(enumerate(dataset)):
        # Load from data
        data = {
            col: torch.tensor(row[i] + 1, dtype=torch.int)
            if i < len(["answerCode"] + args.cate_cols)
            else torch.tensor(row[i] + 1 - min(row[i]), dtype=torch.float).view(-1, 1)
            for i, col in enumerate(["answerCode"] + args.cate_cols + args.cont_cols)
        }
        # Generate mask: max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        seq_len = len(row[0])
        if seq_len > max_seq_len:
            for k, seq in data.items():
                data[k] = seq[-max_seq_len:]
            mask = torch.ones(max_seq_len, dtype=torch.int16)
            position = torch.tensor(
                [max_seq_len + i for i in range(1, max_seq_len + 1)],
                dtype=torch.int,
            )
        else:
            for k, seq in data.items():
                # Pre-padding non-valid sequences
                tmp = torch.zeros([max_seq_len, 1][: len(data[k].size())])
                tmp[max_seq_len - seq_len :] = data[k]
                data[k] = tmp
            mask = torch.zeros(max_seq_len, dtype=torch.int16)
            mask[-seq_len:] = 1
        position = torch.tensor(
            [i for i in range(1, max_seq_len + 1)], dtype=torch.int16
        )
        position = position * mask
        data["Position"] = position
        data["mask"] = mask
        interaction = data["answerCode"]
        interaction = interaction.roll(shifts=1)
        interaction_mask = data["mask"].roll(shifts=1)
        interaction_mask[0] = 0
        interaction = (interaction * interaction_mask).to(torch.int64)
        data["Interaction"] = interaction
        data = {k: v.int() for k, v in data.items()}
        dataset[i] = data
    return dataset
