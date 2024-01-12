import pandas as pd

def make_valid():
    test_df = pd.read_csv('../../data/test_data.csv')
    valid_df = test_df[test_df['answerCode'] != -1]

    valid_df.to_csv('../../data/valid_data.csv', index=False)

def make_full_train():
    train_df = pd.read_csv('../../data/train_data.csv')
    valid_df = pd.read_csv('../../data/valid_data.csv')

    full_train_df = pd.concat([train_df, valid_df], ignore_index=True)

    full_train_df.to_csv('../../data/full_train_data.csv', index=False)

if __name__ == 'main':
    make_valid()
    make_full_train()