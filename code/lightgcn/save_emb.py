import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

from lightgcn.datasets import prepare_dataset
from lightgcn import trainer
from lightgcn.utils import get_logger, logging_conf, set_seeds

logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    os.makedirs(args.model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Preparing data ...")
    train_data, test_data, n_node, id2index = prepare_dataset(
        device=device, data_dir=args.data_dir, tag=args.tag
    )

    logger.info("Loading Model ...")
    weight: str = os.path.join(args.model_dir, args.model_name)
    model: torch.nn.Module = trainer.build(
        n_node=n_node,
        embedding_dim=args.hidden_dim,
        num_layers=args.n_layers,
        alpha=args.alpha,
        weight=weight,
    )
    model = model.to(device)

    out = model.get_embedding(train_data["edge"])
    
    user_emb = out[train_data["edge"][0]]
    item_emb = out[train_data["edge"][1]]
    
    pd.DataFrame(user_emb.cpu().detach().numpy()).to_csv('user_emb_df.csv', index=False)
    pd.DataFrame(item_emb.cpu().detach().numpy()).to_csv('item_emb_df.csv', index=False)


if __name__ == "__main__":
    main()
