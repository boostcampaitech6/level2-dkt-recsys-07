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
    return

    bigtag = np.array(list(map(lambda r: int(r[2]), train_df['assessmentItemID'].unique())))

    fig = plt.figure(figsize=(8,8))
    if n == 2:
        ax = fig.add_subplot(111)
        
        ax.scatter(x_embeded[:,0],
                    x_embeded[:,1],
                    c = (bigtag-1)/9,
                    s = 10,
                    cmap = 'tab10',
                    alpha=0.5)
    elif n == 3:
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_embeded[:,0],
                    x_embeded[:,1],
                    x_embeded[:,2],
                    c = (bigtag-1)/9,
                    s = 10,
                    cmap = 'tab10',
                    alpha=0.5)
    fig.colorbar(plt.cm.ScalarMappable(cmap='tab10'), ax = ax)

    plt.savefig(f"{n}d_tsne.png")

if __name__ == '__main__':
    main()