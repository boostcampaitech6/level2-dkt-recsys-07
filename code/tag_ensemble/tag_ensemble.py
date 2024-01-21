import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args: DictConfig):
    OmegaConf.set_struct(args, False)

    test_name = args.data_dir + args.test_file_name
    test_df = pd.read_csv(test_name)
    test_tag = test_df["assessmentItemID"].apply(lambda x: x[2])
    tag_list = test_tag[test_df["answerCode"] == -1].reset_index()["assessmentItemID"]

    out_df = pd.DataFrame({"id": range(0, 744)})
    out_df["prediction"] = [0.0] * 744

    for i in range(1, 10):
        out_name = args.out_dir + args.out_name + "_tag" + str(i) + ".csv"
        temp_df = pd.read_csv(out_name)
        out_df[tag_list == str(i)] = temp_df[tag_list == str(i)]

    out_df.to_csv(args.out_dir + args.out_name + "_tag_ensemble.csv", index=False)


if __name__ == "__main__":
    main()
