import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    if args.scheduler == "CAWR":
        scheduler = CosineAnnealingWarmRestarts(optimizer, args.patience, verbose=True)
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    return scheduler
