import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from torch_geometric.nn.models import LightGCN
import wandb
from collections import defaultdict

from tqdm import tqdm
from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def build(n_node: int, weight: str = None, **kwargs):
    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def run(
    model: nn.Module,
    train_data: dict,
    valid_data: dict = None,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    model_dir: str = None,
    model_filename: str = "lightgcn",
    id2index: dict = "",
    args="",
):
    model.train()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    os.makedirs(name=model_dir, exist_ok=True)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        valid_len = int(len(train_data["label"]) * 0.05)
        eid_pos = np.where(train_data["label"].cpu().numpy() == 1)[0]
        eid_neg = np.where(train_data["label"].cpu().numpy() == 0)[0]

        eids_pos_list = np.random.RandomState(seed=args.seed).permutation(eid_pos)
        eids_neg_list = np.random.RandomState(seed=args.seed).permutation(eid_neg)

        eids = np.concatenate(
            (eids_pos_list[:valid_len], eids_neg_list[:valid_len]), axis=0
        )
        not_eids = np.concatenate(
            (eids_pos_list[valid_len:], eids_neg_list[valid_len:]), axis=0
        )

        edge, label = train_data["edge"], train_data["label"]
        valid_label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=valid_label[eids])
        train_data = dict(edge=edge[:, not_eids], label=label[not_eids])
        if args.rec_loss:
            label = train_data["label"]
            edge_index = train_data["edge"]

            # 사용자별 양성 및 음성 상호작용 인덱스 구분
            user_pos_interactions = defaultdict(list)
            user_neg_interactions = defaultdict(list)

            for idx, (user_id, item_id) in enumerate(edge_index.t()):
                if label[idx] == 1:
                    user_pos_interactions[user_id.item()].append(idx)
                else:
                    user_neg_interactions[user_id.item()].append(idx)

            # 사용자별로 매칭된 양성 및 음성 상호작용 생성
            matched_pos_indices = []
            matched_neg_indices = []

            for user_id in tqdm(user_pos_interactions.keys()):
                if user_id in user_neg_interactions:
                    # 사용자별로 양성 및 음성 상호작용의 최소 길이 찾기
                    min_length = min(
                        len(user_pos_interactions[user_id]),
                        len(user_neg_interactions[user_id]),
                    )

                    # 사용자별 매칭된 상호작용 추가
                    matched_pos_indices.extend(
                        user_pos_interactions[user_id][:min_length]
                    )
                    matched_neg_indices.extend(
                        user_neg_interactions[user_id][:min_length]
                    )

    logger.info(f"Training Started : n_epochs={n_epochs}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epochs):
        logger.info("Epoch: %s", e)
        # TRAIN
        train_auc, train_acc, train_loss = train(
            train_data=train_data,
            model=model,
            optimizer=optimizer,
            id2index=id2index,
            args=args,
            matched_pos_indices=matched_pos_indices,
            matched_neg_indices=matched_neg_indices,
        )

        # VALID
        auc, acc = validate(valid_data=valid_data, model=model)
        wandb.log(
            dict(
                train_loss_epoch=train_loss,
                train_acc_epoch=train_acc,
                train_auc_epoch=train_auc,
                valid_acc_epoch=acc,
                valid_auc_epoch=auc,
                valid_auc_best= max(auc, best_auc),
            )
        )
        if auc > best_auc:
            logger.info("Best model updated AUC from %.4f to %.4f", best_auc, auc)
            best_auc, best_epoch = auc, e
            if model_filename == "lightgcn":
                torch.save(
                    obj={"model": model.state_dict(), "epoch": e + 1},
                    f=os.path.join(model_dir, f"best_model.pt"),
                )
            else:
                torch.save(
                    obj={"model": model.state_dict(), "epoch": e + 1},
                    f=os.path.join(model_dir, "best_" + model_filename),
                )
    if model_filename == "lightgcn":
        torch.save(
            obj={"model": model.state_dict(), "epoch": e + 1},
            f=os.path.join(model_dir, f"last_model.pt"),
        )
    else:
        torch.save(
            obj={"model": model.state_dict(), "epoch": e + 1},
            f=os.path.join(model_dir, "last_" + model_filename),
        )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def train(
    model: nn.Module,
    train_data: dict,
    optimizer: torch.optim.Optimizer,
    id2index: dict,
    args,
    matched_pos_indices="",
    matched_neg_indices="",
):
    pred = model(train_data["edge"])

    if args.roc_star:
        prob = model.predict_link(edge_index=train_data["edge"], prob=True)
        label = train_data["label"]
        # prob = prob.detach().cpu()
        # label = train_data["label"].cpu()
        loss = roc_star_paper(prob, label, args)
    elif args.rec_loss:
        # 모델을 사용하여 매칭된 양성 및 음성 에지에 대한 예측 점수 계산.
        edge_index = train_data["edge"]
        pos_edge_rank = model.predict_link(
            edge_index[:, matched_pos_indices], prob=True
        )
        neg_edge_rank = model.predict_link(
            edge_index[:, matched_neg_indices], prob=True
        )

        # recommendation_loss 함수를 사용하여 손실 계산
        loss = model.recommendation_loss(
            pos_edge_rank, neg_edge_rank, lambda_reg=args.lambda_reg
        ) + args.loss_factor * model.link_pred_loss(
            pred=pred, edge_label=train_data["label"]
        )
    else:
        loss = model.link_pred_loss(pred=pred, edge_label=train_data["label"])

    prob = model.predict_link(edge_index=train_data["edge"], prob=True)
    prob = prob.detach().cpu().numpy()

    label = train_data["label"].cpu().numpy()
    acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
    auc = roc_auc_score(y_true=label, y_score=prob)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    logger.info("TRAIN AUC : %.4f ACC : %.4f LOSS : %.4f", auc, acc, loss.item())
    return auc, acc, loss


def validate(valid_data: dict, model: nn.Module):
    with torch.no_grad():
        prob = model.predict_link(edge_index=valid_data["edge"], prob=True)
        prob = prob.detach().cpu().numpy()

        label = valid_data["label"]
        acc = accuracy_score(y_true=label, y_pred=prob > 0.5)
        auc = roc_auc_score(y_true=label, y_score=prob)
    logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc


def inference(model: nn.Module, data: dict, output_dir: str):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(edge_index=data["edge"], prob=True)

    logger.info("Saving Result ...")
    pred = pred.detach().cpu().numpy()
    os.makedirs(name=output_dir, exist_ok=True)
    write_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
    logger.info("Successfully saved submission as %s", write_path)


def roc_star_paper(y_pred, _y_true, args):
    batch_size = 2048
    y_true = _y_true >= 0.5

    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]:
        return y_pred.shape[0] * 1e-8
    # 모델의 예측값에 따라 양성과 음성 샘플 분리
    pos_indices = y_true == 1
    neg_indices = y_true == 0

    pos_preds = y_pred[pos_indices]
    neg_preds = y_pred[neg_indices]

    # 음성 샘플 중에서 하드 네거티브 샘플 선택
    # 예: 모델이 양성이라고 예측한 점수가 높은 상위 10%의 음성 샘플 선택
    num_hard_negatives = int(0.05 * len(neg_preds))  # 상위 10%
    hard_negatives_indices = neg_preds.topk(num_hard_negatives).indices

    # 양성 샘플 중에서 하드 포지티브 샘플 선택
    # 예: 모델이 양성이라고 예측한 점수가 낮은 하위 10%의 양성 샘플 선택
    num_hard_positives = int(0.05 * len(pos_preds))  # 하위 10%
    hard_positives_indices = pos_preds.topk(num_hard_positives, largest=False).indices

    # 하드 네거티브 샘플을 사용하여 학습 데이터셋 구성
    train_pos_preds = pos_preds[hard_positives_indices]
    train_neg_preds = neg_preds[hard_negatives_indices]
    # pos = y_pred[y_true]
    # neg = y_pred[~y_true]

    total_loss = 0.0
    num_batches = 0

    # 배치 처리
    for pos_batch in tqdm(torch.split(train_pos_preds, batch_size)):
        for neg_batch in torch.split(train_neg_preds, batch_size):
            ln_pos = pos_batch.shape[0]
            ln_neg = neg_batch.shape[0]

            # 필터링
            pos_batch_filtered = pos_batch[pos_batch < max(neg_batch) + args.gamma]
            neg_batch_filtered = neg_batch[neg_batch > min(pos_batch) - args.gamma]

            # 확장과 차이 계산
            pos_expand = pos_batch_filtered.view(-1, 1).expand(-1, ln_neg).reshape(-1)
            neg_expand = neg_batch_filtered.repeat(ln_pos)
            diff = -(pos_expand - neg_expand - args.gamma)
            diff = diff[diff > 0]

            # 손실 계산
            loss = torch.sum(diff * diff)
            loss = loss / (ln_pos + ln_neg)

            total_loss += loss
            num_batches += 1

    # 평균 손실 반환
    avg_loss = total_loss / num_batches if num_batches > 0 else y_pred.shape[0] * 1e-8
    return avg_loss + 1e-8
