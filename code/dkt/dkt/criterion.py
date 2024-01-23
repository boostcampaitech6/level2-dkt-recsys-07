import torch
from torchvision.ops import sigmoid_focal_loss

def get_criterion(pred: torch.Tensor, target: torch.Tensor, args):
    if args.loss_function.lower() == "bce":
        loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(pred, target)
    elif args.loss_function.lower() == "focal":
        loss = sigmoid_focal_loss(pred, target, args.focal_alpha, args.focal_gamma)
    return loss

