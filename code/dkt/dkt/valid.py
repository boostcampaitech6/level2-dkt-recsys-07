import pandas as pd

def make_valid():
    test_df = pd.read_csv('../../data/test_data.csv')
    valid_df = test_df[test_df['answerCode'] != -1]

    valid_df.to_csv('../../data/valid_data.csv')