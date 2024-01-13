import os

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, logging_conf


logger = get_logger(logging_conf)


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    os.makedirs(args.model_dir, exist_ok=True)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Preparing data ...")
    preprocess = Preprocess(args=args)
    preprocess.load_test_data(file_name=args.test_file_name)
    test_data: np.ndarray = preprocess.get_test_data()

    logger.info("Loading Model ...")
    model: torch.nn.Module = trainer.load_model(args=args).to(args.device)
    for k, v in model.embedding_dict.items():
        v.to(args.device)

    logger.info("Make Predictions & Save Submission ...")
    trainer.inference(args=args, test_data=test_data, model=model)


if __name__ == "__main__":
    main()
