import pandas as pd
import numpy as np
import ot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances


def load_data():
    source = pd.read_csv("data/source.csv")
    target = pd.read_csv("data/target.csv")

    Xs = source.drop("label", axis=1).values
    ys = source["label"].values

    Xt = target.drop("label", axis=1).values
    yt = target["label"].values  # only for final evaluation

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xs)
    Xt = scaler.transform(Xt)

    return Xs, ys, Xt, yt

def align_domains_global_sinkhorn(Xs, Xt, reg=1.0):
    ns, nt = Xs.shape[0], Xt.shape[0]

    a = np.ones(ns) / ns
    b = np.ones(nt) / nt

    M = cosine_distances(Xs, Xt)
    M = M / (M.max() + 1e-8)

    G = ot.sinkhorn(
        a, b, M, reg,
        numItermax=1000,
        stopThr=1e-6
    )

    return G @ Xt

def align_domains_classwise_sinkhorn(Xs, ys, Xt, yt, reg=1.0):
    Xs_aligned = np.zeros_like(Xs)

    for c in np.unique(ys):
        Xs_c = Xs[ys == c]
        Xt_c = Xt[yt == c]

        if len(Xs_c) == 0 or len(Xt_c) == 0:
            continue

        ns = Xs_c.shape[0]
        nt = Xt_c.shape[0]

        a = np.ones(ns) / ns
        b = np.ones(nt) / nt

        M = cosine_distances(Xs_c, Xt_c)
        M = M / (M.max() + 1e-8)

        G = ot.sinkhorn(a, b, M, reg)

        Xs_aligned[ys == c] = G.dot(Xt_c)

    return Xs_aligned