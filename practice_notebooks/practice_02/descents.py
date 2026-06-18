import numpy as np
from abc import ABC, abstractmethod
from interfaces import LearningRateSchedule, AbstractOptimizer, LinearRegressionInterface


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 0.1):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        # TODO: реализовать формулу затухающего шага обучения
        lr = self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p
        return lr


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """
    def __init__(self, 
                 lr_schedule: LearningRateSchedule = TimeDecayLR(), 
                 tolerance: float = 1e-6,
                 max_iter: int = 1000
                ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        # TODO: implement
        # в конце также приcваивает атрибуту модели полученный loss_history
        loss_history = []
        init_loss = self.model.compute_loss()
        loss_history.append(init_loss)
        for _ in range(self.max_iter):
            weight_diff = self._step()
            concurrent_loss = self.model.compute_loss()
            loss_history.append(concurrent_loss)
            if np.any(np.isnan(weight_diff)):
                break
            if np.dot(weight_diff, weight_diff) < self.tolerance:
                break
        self.model.loss_history = loss_history


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать vanilla градиентный спуск
        # Можно использовать атрибуты класса self.model
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        delta_w = -lr * gradient
        self.model.w += delta_w
        return delta_w


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать стохастический градиентный спуск
        # 1) выбрать случайный батч
        # 2) вычислить градиенты на батче
        # 3) обновить веса модели
        n_obj = self.model.X_train.shape[0]
        batch_ind = np.random.randint(0, n_obj, size = self.batch_size)
        X_batch = self.model.X_train[batch_ind]
        y_batch = self.model.y_train[batch_ind]
        gradient = self.model.compute_gradients(X_batch, y_batch)
        lr = self.lr_schedule.get_lr(self.iteration)
        delta_w = -lr * gradient
        self.model.w += delta_w
        return delta_w


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать SAG
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            # TODO: инициализировать хранилища при первом вызове
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)
        # TODO: реализовать SAG
        batch_inds = np.random.randint(0, num_objects, size=self.batch_size)
        for i in batch_inds:
            new_g_i = self.model.compute_gradients(X_train[i:i+1], y_train[i:i+1])
            old_g_i = self.grad_memory[i]
            self.grad_sum = self.grad_sum - old_g_i + new_g_i
            self.grad_memory[i] = new_g_i
        avg_grad = self.grad_sum / num_objects
        lr = self.lr_schedule.get_lr(self.iteration)
        delta_w = -lr*avg_grad
        self.model.w += delta_w
        return delta_w


class MomentumDescent(BaseDescent):
    def __init__(self,  *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать градиентный спуск с моментумом
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        self.velocity = self.beta * self.velocity + lr * gradient
        delta_w = -self.velocity
        self.model.w += delta_w
        return delta_w


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        # TODO: реализовать Adam по формуле из ноутбука
        if self.m is None or self.v is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)
        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train)
        k = self.iteration + 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** k)
        v_hat = self.v / (1 - self.beta2 ** k)
        lr = self.lr_schedule.get_lr(self.iteration)
        delta_w = -(lr * m_hat) / (np.sqrt(v_hat) + self.eps)
        self.model.w += delta_w
        return delta_w


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений 
    """
    def __init__(self):
        self.model = None
    

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        X = self.model.X_train
        y = self.model.y_train
        opt_w = self.model.loss_function.analytic_solution(X, y)
        self.model.w = opt_w