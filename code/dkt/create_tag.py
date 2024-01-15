import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)
    train_name = args.data_dir + args.file_name
    test_name = args.data_dir + args.test_file_name
    train_df = pd.read_csv(train_name)
    test_df = pd.read_csv(test_name)

    train_tag = train_df["assessmentItemID"].apply(lambda x: x[2])
    test_tag = test_df["assessmentItemID"].apply(lambda x: x[2])

    for i in [str(i) for i in range(1, 10)]:
        tag_train_df = train_df[train_tag == i]
        tag_test_df = test_df[test_tag == i]

        tag_train_df.to_csv(
            args.data_dir + "train_data_tag" + str(i) + ".csv", index=False
        )
        tag_test_df.to_csv(
            args.data_dir + "test_data_tag" + str(i) + ".csv", index=False
        )


if __name__ == "__main__":
    main()
