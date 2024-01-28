import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering, AffinityPropagation, MeanShift, estimate_bandwidth

import warnings
warnings.filterwarnings('ignore')

def main():
    n = 2
    df = pd.read_csv('tsne_df.csv')

    train_df = pd.read_csv('../../data/full_train_data.csv')
    bigtag = np.array(list(map(lambda r: int(r[2]), train_df['assessmentItemID'].unique())))
    order = np.array(list(map(lambda r: int(r[7:]), train_df['assessmentItemID'].unique())))

    x_min, x_max = min(df.iloc[:,0]), max(df.iloc[:,1])
    y_min, y_max = min(df.iloc[:,1]), max(df.iloc[:,1])
    if n == 2:
        fig1 = plt.figure(1, figsize = (32,32))
        for i in range(1,10):
            ax = fig1.add_subplot(3, 3, i)
            ax.scatter(df.iloc[:,0][bigtag == i],
                        df.iloc[:,1][bigtag == i],
                        s = 10,
                        )
            ax.set_title(f'BigTag {i}', fontsize = 40)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig1.savefig(f"{n}d_cluster.png")
        plt.close(fig1)

        fig2 = plt.figure(2, figsize = (24,24))
        ax = fig2.add_subplot(111)
        ax.scatter(df.iloc[:,0],
                    df.iloc[:,1],
                    c = bigtag/9,
                    s = 100,
                    cmap = 'tab10',
                    alpha=0.5)
        fig2.colorbar(plt.cm.ScalarMappable(cmap='tab10'), ax = ax)
        ax.set_title('BigTag', fontsize = 60)
        
        fig2.savefig(f"{n}d_cluster_full.png")
        plt.close(fig2)

        fig3 = plt.figure(3, figsize = (40,24))
        for i in range(1,14):
            ax = fig3.add_subplot(3, 5, i)
            ax.scatter(df.iloc[:,0][order == i],
                        df.iloc[:,1][order == i],
                        s = 10,
                        )
            ax.set_title(f'Order {i}', fontsize = 40)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        fig3.savefig(f"{n}d_cluster_order.png")
        plt.close(fig3)

        fig4 = plt.figure(4, figsize = (24,24))
        ax = fig4.add_subplot(111)
        ax.scatter(df.iloc[:,0],
                    df.iloc[:,1],
                    c = order/13,
                    s = 100,
                    cmap = 'tab20',
                    alpha=0.5)
        fig4.colorbar(plt.cm.ScalarMappable(cmap='tab20'), ax = ax)
        ax.set_title('Order', fontsize = 60)

        fig4.savefig(f"{n}d_cluster_full_order.png")
        plt.close(fig4)
    elif n == 3:
        return
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(d1.iloc[:,0],
                    d1.iloc[:,1],
                    d1.iloc[:,2],
                    c = bigtag/9,
                    s = 1,
                    cmap = 'tab10',
                    alpha=0.5)
        ax.scatter(cluster.cluster_centers_[:,0], cluster.cluster_centers_[:,1], cluster.cluster_centers_[:,2], c='black', marker='*')
    #fig.colorbar(plt.cm.ScalarMappable(cmap='tab10'), ax = axes[:,:])


if __name__ == '__main__':
    main()