import math
import os

import numpy as np
import torch
from torch import nn, sigmoid
from torch_geometric.nn.models import LightGCN
import wandb

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import (
    LSTM,
    LSTMATTN,
    BERT,
    LastQueryTransformerEncoderLSTM,
    TransformerEncoderLSTM,
    VanillaLQTL,
    GCNLSTM,
    GCNLSTMATTN,
    GCNLastQueryTransformerEncoderLSTM,
    GCNTransformerEncoderLSTM,
    MF,
    LMF,
)
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


def run(args, train_data: np.ndarray, valid_data: np.ndarray, model: nn.Module):
    train_loader, valid_loader = get_loaders(
        args=args, train=train_data, valid=valid_data
    )

    # For warmup scheduler which uses step interval
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        logger.info("Start Training: Epoch %s", epoch + 1)

        # TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
        )

        # VALID
        auc, acc, loss = validate(valid_loader=valid_loader, model=model, args=args)

        wandb.log(
            dict(
                epoch=epoch,
                train_loss_epoch=train_loss,
                train_auc_epoch=train_auc,
                train_acc_epoch=train_acc,
                valid_loss_epoch=loss,
                valid_auc_epoch=auc,
                valid_acc_epoch=acc,
                best_valid_auc=max(best_auc, auc),
            )
        )

        if auc > best_auc:
            best_auc = auc
            # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            if args.cv > 0:
                model_filename = f"{args.model.name}_CV_{args.cv}_{args.model_name}"
            else:
                model_filename = f"{args.model.name}_{args.model_name}"
            save_checkpoint(
                state={"epoch": epoch + 1, "state_dict": model_to_save.state_dict()},
                model_dir=args.model_dir,
                model_filename=model_filename,
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                logger.info(
                    "EarlyStopping counter: %s out of %s",
                    early_stopping_counter,
                    args.patience,
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)


def train(
    train_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args,
):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        if args.model.name.lower() in ["mf", "lmf"]:
            batch = batch[0].to(args.device)
            # loss 계산을 위해 shape 변경
            preds = model(batch[:, :2]).unsqueeze(1)
            targets = batch[:, -1].unsqueeze(1)
        else:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            preds = model(**batch)
            targets = batch["answerCode"] - 1

        if args.roc_star == True:
            loss = roc_star_paper(preds[:, -1], targets[:, -1], args)
        else:
            loss = compute_loss(preds=preds, targets=targets, args=args)
        update_params(
            loss=loss, model=model, optimizer=optimizer, scheduler=scheduler, args=args
        )

        if step % args.log_steps == 0:
            logger.info("Training steps: %s Loss: %.4f", step, loss.item())

        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
    return auc, acc, loss_avg


def validate(valid_loader: nn.Module, model: nn.Module, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(valid_loader):
        if args.model.name.lower() in ["mf", "lmf"]:
            batch = batch[0].to(args.device)
            preds = model(batch[:, :2]).unsqueeze(1)
            targets = batch[:, -1].unsqueeze(1)
        else:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            preds = model(**batch)
            targets = batch["answerCode"] - 1

        losses.append(compute_loss(preds=preds, targets=targets, args=args))
        # predictions
        preds = sigmoid(preds[:, -1])
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    loss_avg = sum(losses) / len(losses)
    logger.info("VALID AUC : %.4f ACC : %.4f loss: %.4f", auc, acc, loss_avg)
    return auc, acc, loss_avg


def inference(args, test_data: np.ndarray, model: nn.Module) -> None:
    model.eval()
    _, test_loader = get_loaders(args=args, train=None, valid=test_data)

    total_preds = []
    for step, batch in enumerate(test_loader):
        if args.model.name.lower() in ["mf", "lmf"]:
            batch = batch[0].to(args.device)
            preds = model(batch[:, :2]).unsqueeze(1)
        else:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            preds = model(**batch)

        # predictions
        preds = sigmoid(preds[:, -1])
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    if args.cv > 0:
        output_file_name = f"{args.model.name}_CV_{args.cv}_submission.csv"
    else:
        output_file_name = f"{args.model.name}_submission.csv"
    write_path = os.path.join(args.output_dir, output_file_name)
    os.makedirs(name=args.output_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


def gcn_build(n_node: int, weight: str = None, **kwargs):
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


def get_model(args) -> nn.Module:
    try:
        model_name = args.model.name.lower()
        model = {
            "lstm": LSTM,
            "lstmattn": LSTMATTN,
            "bert": BERT,
            "lqtl": LastQueryTransformerEncoderLSTM,
            "tl": TransformerEncoderLSTM,
            "vlqtl": VanillaLQTL,
            "gcnlstm": GCNLSTM,
            "gcnlstmattn": GCNLSTMATTN,
            "gcnlqtl": GCNLastQueryTransformerEncoderLSTM,
            "gcntl": GCNTransformerEncoderLSTM,
            "mf": MF,
            "lmf": LMF,
        }.get(model_name)(args)
        if args.model.name[:3].lower() == "gcn":
            weight: str = os.path.join(
                args.model.gcn.model_dir, args.model.gcn.model_name
            )
            gcn_model: torch.nn.Module = gcn_build(
                n_node=args.model.gcn.n_node,
                embedding_dim=args.model.gcn.hidden_dim,
                num_layers=args.model.gcn.n_layers,
                alpha=args.model.gcn.alpha,
                weight=weight,
            )
            model.gcn_embedding.weight = nn.Parameter(
                gcn_model.embedding.weight.clone(), requires_grad=False
            )

    except KeyError:
        logger.warn("No model name %s found", model_name)
    except Exception as e:
        logger.warn("Error while loading %s with args: %s", model_name, args)
        raise e
    return model


def compute_loss(preds: torch.Tensor, targets: torch.Tensor, args):
    """
    loss계산하고 parameter update
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(pred=preds, target=targets.float(), args=args)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    args,
):
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state: dict, model_dir: str, model_filename: str) -> None:
    """Saves checkpoint to a given directory."""
    save_path = os.path.join(model_dir, model_filename)
    logger.info("saving model as %s...", save_path)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(state, save_path)


def load_model(args):
    if args.cv > 0:
        model_name = f"{args.model.name}_CV_{args.cv}_{args.model_name}"
    else:
        model_name = f"{args.model.name}_{args.model_name}"
    model_path = os.path.join(args.model_dir, model_name)
    logger.info("Loading Model from: %s", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)
    logger.info("Successfully loaded model state from: %s", model_path)
    return model


def roc_star_paper(y_pred, _y_true, args):
    y_true = _y_true >= 0.5

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]:
        return y_pred.shape[0] * 1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    max_pos = 1000 # Max number of positive training samples
    max_neg = 1000 # Max number of positive training samples
    pos = pos[torch.rand_like(pos) < max_pos/ln_pos]
    neg = neg[torch.rand_like(neg) < max_neg/ln_neg]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    
    pos_expand = pos.view(-1,1).expand(-1,ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)

    diff = -(pos_expand - neg_expand - args.gamma)
    diff = diff[diff > 0]

    loss = torch.sum(diff * diff)
    loss = loss / (ln_pos + ln_neg)
    return loss + 1e-8
