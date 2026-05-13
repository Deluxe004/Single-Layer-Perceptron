import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#Настройка стиля для графиков
plt.style.use('ggplot')

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

#Класс перцептрона
class SingleLayerPerceptron:
    def __init__(self, input_dim: int, init_type: str = 'small'):
        if init_type == 'zeros':
            self.w = np.zeros(input_dim)
        elif init_type == 'large':
            self.w = np.random.normal(0, 10.0, input_dim)
        else:
            self.w = np.random.normal(0, 0.01, input_dim)
        self.b = 0.0
        self.train_loss_history, self.val_loss_history = [], []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size,
            loss_type='bce', l2_lambda=0.0, beta=0.0):
        m = X_train.shape[0]
        self.train_loss_history, self.val_loss_history = [], []
        v_w, v_b = np.zeros_like(self.w), 0.0

        for epoch in range(epochs):
            idx = np.random.permutation(m)
            X_s, y_s = X_train[idx], y_train[idx]
            #Разбиение на батчи
            for i in range(0, m, batch_size):
                X_b, y_b = X_s[i:i + batch_size], y_s[i:i + batch_size]
                #Вычис-е градиентов
                if loss_type == 'bce':  #Доп задание 2 (L2)
                    err = self.forward(X_b) - y_b
                    dw = (1 / len(y_b)) * (X_b.T @ err) + 2 * l2_lambda * self.w
                    db = (1 / len(y_b)) * np.sum(err)
                elif loss_type == 'hinge':  #Доп задание 2 (Hinge)
                    y_b_hinge = np.where(y_b == 0, -1, 1)
                    z = X_b @ self.w + self.b
                    mask = (1 - y_b_hinge * z) > 0
                    dw = -(1 / len(y_b)) * (X_b.T @ (y_b_hinge * mask)) + 2 * l2_lambda * self.w
                    db = -(1 / len(y_b)) * np.sum(y_b_hinge * mask)

                #Доп задание 4 (Momentum SGD, работает если beta > 0)
                v_w = beta * v_w + dw
                v_b = beta * v_b + db
                self.w -= lr * v_w
                self.b -= lr * v_b

            #Логирование
            if loss_type == 'bce':
                self.train_loss_history.append(
                    self.compute_loss(y_train, self.forward(X_train)) + l2_lambda * np.sum(self.w ** 2))
                self.val_loss_history.append(
                    self.compute_loss(y_val, self.forward(X_val)) + l2_lambda * np.sum(self.w ** 2))
            else:
                y_tr_h, y_val_h = np.where(y_train == 0, -1, 1), np.where(y_val == 0, -1, 1)
                self.train_loss_history.append(
                    np.mean(np.maximum(0, 1 - y_tr_h * (X_train @ self.w + self.b))) + l2_lambda * np.sum(self.w ** 2))
                self.val_loss_history.append(
                    np.mean(np.maximum(0, 1 - y_val_h * (X_val @ self.w + self.b))) + l2_lambda * np.sum(self.w ** 2))

    def predict(self, X: np.ndarray, is_hinge: bool = False) -> np.ndarray:
        z = X @ self.w + self.b
        if is_hinge:
            #Для Hinge Loss предсказываем по знаку сырого значения (без сигмоиды)
            return (z >= 0).astype(int)
        else:
            #Для стандартного перцептрона предсказываем по вероятности
            return (self.sigmoid(z) >= 0.5).astype(int)


#Доп функции генератор и метрики
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

def compute_metrics(y_true, y_pred):
    tp, fp = np.sum((y_pred == 1) & (y_true == 1)), np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec, rec = tp / (tp + fp + 1e-15), tp / (tp + fn + 1e-15)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec + 1e-15)}

def compute_roc_auc(y_true, y_prob):
    desc_idx = np.argsort(y_prob)[::-1]
    y_sorted = y_true[desc_idx]
    pos_total, neg_total = np.sum(y_true == 1), np.sum(y_true == 0)
    tpr_list, fpr_list, tp, fp = [0.0], [0.0], 0, 0
    for yt in y_sorted:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / pos_total)
        fpr_list.append(fp / neg_total)
    tpr, fpr = np.array(tpr_list), np.array(fpr_list)
    return fpr, tpr, np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)


#Запуск экспериментов
if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_lab_data()

    print("Эксперименты 1-4 (Обязательная часть)")
    # Базовое обучение
    model = SingleLayerPerceptron(input_dim=2)
    model.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
    print(f"Базовая модель; Точность Test: {np.mean(model.predict(X_test) == y_test):.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', alpha=0.7, edgecolors='k')
    x1 = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
    x2 = -(model.w[0] * x1 + model.b) / (model.w[1] + 1e-8)  # Защита + 1e-8
    plt.plot(x1, x2, 'k-', lw=2, label='Разделяющая прямая')
    plt.title("Эксперимент 1: Разделяющая граница (Базовая модель)")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    def run_param_experiment(param_name: str, values: list, title: str):
        plt.figure(figsize=(7, 4))
        for val in values:
            lr, bs, init = 0.1, 32, 'small'
            if param_name == "lr":
                lr = val
            elif param_name == "batch_size":
                bs = val
            elif param_name == "init_type":
                init = val

            mdl = SingleLayerPerceptron(2, init_type=init)
            mdl.fit(X_train, y_train, X_val, y_val, epochs=100, lr=lr, batch_size=bs)

            if val == 'zeros':
                plt.plot(mdl.val_loss_history, label=f"{param_name} = {val}", lw=5, ls='--', alpha=0.5, c='blue')
            else:
                plt.plot(mdl.val_loss_history, label=f"{param_name} = {val}", lw=2)

        plt.title(f"Влияние {title}")
        plt.xlabel("Эпохи")
        plt.ylabel("Функция потерь")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    run_param_experiment("lr", [0.001, 0.01, 0.5, 1.0], "скорости обучения (LR)")
    run_param_experiment("batch_size", [1, 16, 64, 256], "размера батча")

    plt.figure(figsize=(8, 5))
    init_types = ['zeros', 'small', 'large']

    for val in init_types:
        mdl = SingleLayerPerceptron(2, init_type=val)
        mdl.fit(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32)
        acc = np.mean(mdl.predict(X_test) == y_test)
        print(f"Инициализация: {val:<6} | Точность (Test): {acc:.4f} | Итоговый Loss: {mdl.val_loss_history[-1]:.4f}")

        if val == 'zeros':
            plt.plot(mdl.val_loss_history, label="zeros", lw=6, ls=':', alpha=0.5, c='blue')
        elif val == 'small':
            plt.plot(mdl.val_loss_history, label="small", lw=2, c='red')
        elif val == 'large':
            plt.plot(mdl.val_loss_history, label="large (N(0, 10))", lw=2, c='green')

    plt.title("Эксперимент: Влияние инициализации весов")
    plt.xlabel("Количество эпох")
    plt.ylabel("Функция потерь")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

#Доп задания
    print("Доп задание 1: Границы применимости перцептрон")
    # Создаем 3 графика в ряд
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['linear', 'xor', 'circle']
    titles = ['Линейные (Успех)', 'XOR (Провал)', 'Круг (Провал)']

    for i, mode in enumerate(modes):
        #Генерируем данные
        noise_level = 0.05 if mode == 'linear' else 0.0
        X_cust, y_cust = generate_custom_data(mode, n=400, noise=noise_level)

        #Обучаем отдельную модель для каждого датасета
        mdl_cust = SingleLayerPerceptron(2)
        mdl_cust.fit(X_cust, y_cust, X_cust, y_cust, epochs=100, lr=0.1, batch_size=32)
        acc = np.mean(mdl_cust.predict(X_cust) == y_cust)

        #Рисуем точки
        ax = axes[i]
        ax.scatter(X_cust[:, 0], X_cust[:, 1], c=y_cust, cmap='bwr', alpha=0.7, edgecolors='k')

        #Рисуем разделяющую прямую
        x_min, x_max = X_cust[:, 0].min(), X_cust[:, 0].max()
        x1 = np.linspace(x_min, x_max, 100)
        x2 = -(mdl_cust.w[0] * x1 + mdl_cust.b) / (mdl_cust.w[1] + 1e-8)

        #Ограничиваем ось Y, чтобы кривая линия не ломала масштаб графика
        y_min, y_max = X_cust[:, 1].min(), X_cust[:, 1].max()
        ax.set_ylim([y_min - 0.5, y_max + 0.5])

        ax.plot(x1, x2, 'k-', lw=3, label='Граница')
        ax.set_title(f"{titles[i]}\nТочность: {acc:.2f}")
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show(block=False)

    print("Доп задание 2: Обучение моделей с Hinge Loss и исследование L2")
    #Подготавливаем линейные данные
    X_lin, y_lin = generate_custom_data('linear', 300)
    X_lin = (X_lin - X_lin.mean(axis=0)) / (X_lin.std(axis=0) + 1e-8)

    #Сравнение сходимости
    mdl_bce = SingleLayerPerceptron(2)
    mdl_hinge = SingleLayerPerceptron(2)
    mdl_bce.fit(X_lin, y_lin, X_lin, y_lin, epochs=50, lr=0.01, batch_size=32, loss_type='bce')
    mdl_hinge.fit(X_lin, y_lin, X_lin, y_lin, epochs=50, lr=0.01, batch_size=32, loss_type='hinge')

    #Исследование влияния L2-регуляризации
    lambdas = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]
    weight_norms = []
    acc_vals = []

    for lam in lambdas:
        m_l2 = SingleLayerPerceptron(2)
        m_l2.fit(X_lin, y_lin, X_lin, y_lin, epochs=50, lr=0.01, batch_size=32, loss_type='hinge', l2_lambda=lam)

        #Считаем длину вектора весов (Евклидова норма) - L2 должна её уменьшать
        norm_w = np.sqrt(np.sum(m_l2.w ** 2))
        weight_norms.append(norm_w)
        acc_vals.append(np.mean(m_l2.predict(X_lin, is_hinge=True) == y_lin))

    #График
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #График слева: Сравнение Loss
    ax1.plot(mdl_bce.train_loss_history, label='BCE Loss', color='blue', lw=2)
    ax1.set_ylabel('Binary Cross-Entropy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax1_twin = ax1.twinx()  #Делаем вторую ось Y для Hinge, так как масштаб другой
    ax1_twin.plot(mdl_hinge.train_loss_history, label='Hinge Loss', color='red', lw=2)
    ax1_twin.set_ylabel('Hinge Loss', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    ax1.set_title("Доп 2: Сходимость BCE vs Hinge")
    ax1.set_xlabel("Эпохи")

    #График справа: Влияние L2
    ax2.plot(lambdas, weight_norms, marker='o', color='purple', label='Норма весов ||w||', lw=2)
    ax2.set_ylabel("Норма весов ||w|| (Размер весов)", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    ax2_twin = ax2.twinx()
    ax2_twin.plot(lambdas, acc_vals, marker='s', color='green', label='Точность', linestyle='--')
    ax2_twin.set_ylabel("Точность на тренировке", color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')

    ax2.set_title("Доп 2: Влияние коэффициента L2")
    ax2.set_xlabel("Логарифмический масштаб")
    ax2.set_xscale('symlog', linthresh=0.01)  #Чтобы маленькие лямбды были хорошо видно

    plt.tight_layout()
    plt.show(block=False)

    print("Доп задание 3: ROC AUC и Визуализация ошибок")

    preds, probs = model.predict(X_test), model.forward(X_test)
    met = compute_metrics(y_test, preds)
    fpr, tpr, auc = compute_roc_auc(y_test, probs)
    print(f"Precision: {met['precision']:.3f} | Recall: {met['recall']:.3f} | F1: {met['f1']:.3f} | AUC: {auc:.3f}")

    #Строим 2 графика рядом: Ошибки и ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    #График 1: Ошибки
    correct = (preds == y_test)
    ax1.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Верно', alpha=0.6, edgecolors='k')
    ax1.scatter(X_test[~correct, 0], X_test[~correct, 1], c='red', label='Ошибка', marker='X', s=100)
    ax1.set_title("Доп 3: Анализ ошибок")
    ax1.legend()

    #График 2: ROC-кривая
    ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC-кривая (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax2.set_title("Доп 3: ROC-кривая")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    plt.tight_layout()
    plt.show(block=False)

    print("Доп задание 4: Исследование Momentum SGD")
    plt.figure(figsize=(7, 4))
    for beta in [0.0, 0.5, 0.9, 0.99]:
        m_mom = SingleLayerPerceptron(2)
        m_mom.fit(X_train, y_train, X_val, y_val, epochs=40, lr=0.1, batch_size=32, beta=beta)
        plt.plot(m_mom.train_loss_history, label=f'Импульс β = {beta}')

    plt.title("Доп 4: Ускорение сходимости с Momentum (Train Loss)")
    plt.xlabel("Эпохи")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    print("Доп задание 5: 5-Fold Кросс-валидация")
    cv_folds = get_kfold_indices(n_samples=len(X_train), n_splits=5)  #Делаем CV на Train данных
    best_acc, best_params = 0, None

    for lr in [0.01, 0.1]:
        for bs in [16, 32]:
            fold_accs = []
            for tr_i, va_i in cv_folds:
                X_tr, X_va = X_train[tr_i], X_train[va_i]
                y_tr, y_va = y_train[tr_i], y_train[va_i]

                m_cv = SingleLayerPerceptron(2)
                m_cv.fit(X_tr, y_tr, X_va, y_va, epochs=50, lr=lr, batch_size=bs)
                fold_accs.append(np.mean(m_cv.predict(X_va) == y_va))

            mean_a, std_a = np.mean(fold_accs), np.std(fold_accs)
            print(f"CV -> LR: {lr}, Батч: {bs} | Точность: {mean_a:.4f} +/- {std_a:.4f}")
            if mean_a > best_acc:
                best_acc, best_params = mean_a, (lr, bs)

    print(f"Лучшие параметры по CV: LR = {best_params[0]}, BatchSize = {best_params[1]}")
    plt.show()