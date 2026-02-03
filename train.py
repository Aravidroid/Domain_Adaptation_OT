import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class NearestClassMean:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = {
            c: X[y == c].mean(axis=0)
            for c in self.classes_
        }

    def predict(self, X):
        preds = []
        for x in X:
            dists = {
                c: np.linalg.norm(x - self.means_[c])
                for c in self.classes_
            }
            preds.append(min(dists, key=dists.get))
        return np.array(preds)


def train_ncm(X, y):
    model = NearestClassMean()
    model.fit(X, y)
    return model


def train_knn(X, y, k=1):
    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="euclidean"
    )
    model.fit(X, y)
    return model


def evaluate(model, X, y):
    preds = model.predict(X)
    return accuracy_score(y, preds)
