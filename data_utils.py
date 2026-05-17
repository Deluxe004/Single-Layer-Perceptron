import numpy as np
from sklearn.datasets import make_classification

#Собственные утилиты
def stratified_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42):
    np.random.seed(random_state)
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split_point = int(len(idx) * (1.0 - test_size))
        train_idx.extend(idx[:split_point])
        test_idx.extend(idx[split_point:])
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def get_kfold_indices(n_samples: int, n_splits: int = 5, random_state: int = 42):
    rng = np.random.default_rng(random_state) #Локальный генератор
    indices = rng.permutation(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, test_idx))
        current = stop
    return folds

def get_lab_data():
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.3, random_state=42)

    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1e-8

    X_train_norm = (X_train - mu) / sigma
    X_test_norm = (X_test - mu) / sigma

    #Возвращаем X_test и y_test дважды (и как тест, и как валидацию), чтобы графики работали
    return X_train_norm, y_train, X_test_norm, y_test, X_test_norm, y_test

def generate_custom_data(mode='linear', n=500, noise=0.0):
    np.random.seed(42)
    if mode == 'linear':
        X1 = np.random.randn(n // 2, 2) + [1.5, 1.5]
        X2 = np.random.randn(n // 2, 2) - [1.5, 1.5]
        X, y = np.vstack([X1, X2]), np.array([1] * (n // 2) + [0] * (n // 2))
    elif mode == 'xor':
        X = np.random.uniform(-1, 1, (n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    elif mode == 'circle':
        r = np.sqrt(np.random.uniform(0, 1, n))
        theta = np.random.uniform(0, 2 * np.pi, n)
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        y = (r > 0.7).astype(int)
    if noise > 0:
        flip = np.random.rand(n) < noise
        y[flip] = 1 - y[flip]
    return X, y