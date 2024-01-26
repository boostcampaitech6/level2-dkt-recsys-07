import pandas as pd
from dkt.args import parse_args


def cv_ensemble(args):
    df = pd.DataFrame()

    for i in range(1, 6):
        cv = pd.read_csv(f"outputs/{args.model}_CV_{i}_submission.csv")
        df[f"cv{i}"] = cv["prediction"]

    result = pd.DataFrame()
    result["prediction"] = df.mean(axis="columns")
    result.reset_index(inplace=True)
    result.rename(columns={"index": "id"}, inplace=True)

    result.to_csv(f"outputs/{args.model}_CV_submission.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    cv_ensemble(args)
