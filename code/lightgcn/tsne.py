import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def main():
    n = 2
    df = pd.read_csv('item_emb_df.csv')
    train_df = pd.read_csv('../../data/full_train_data.csv')
    item_idx = train_df.drop_duplicates('assessmentItemID', keep='first').index
    df = df.iloc[item_idx]

    print('df loaded')

    model = TSNE(n_components=n)
    x_embeded = model.fit_transform(df)
    pd.DataFrame(x_embeded).to_csv('tsne_df.csv',index=False)

if __name__ == '__main__':
    main()
