import os

import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from dkt import trainer
from dkt.dataloader import Preprocess, sliding_window
from dkt.utils import get_logger, set_seeds, logging_conf


logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    os.makedirs(args.model_dir, exist_ok=True)
    wandb.login()
    set_seeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Preparing data ...")
    preprocess = Preprocess(args)
    preprocess.load_train_data(file_name=args.file_name)
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data, seed=args.seed)
    train_data = sliding_window(args, train_data)
    wandb.init(
        project="dkt",
        config=OmegaConf.to_container(args, resolve=True, throw_on_missing=True),
    )

    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(args=args).to(args.device)

    logger.info("Start Training ...")
    trainer.run(args=args, train_data=train_data, valid_data=valid_data, model=model)


if __name__ == "__main__":
    main()
