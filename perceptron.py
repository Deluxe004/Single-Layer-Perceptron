import numpy as np

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