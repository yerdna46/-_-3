import numpy as np
from scipy.optimize import minimize

# Функция Розенброка
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

# Градиент функции Розенброка
def rosenbrock_grad(x):
    grad = np.zeros_like(x)
    grad[0:-1] = -400.0*(x[1:]-x[:-1]**2.0)*x[:-1] - 2.0*(1-x[:-1])
    grad[1:] += 200.0*(x[1:]-x[:-1]**2.0)
    return grad

# Градиентный спуск
def gradient_descent(grad_func, start_point, learning_rate=0.001, max_iter=1000):
    x = start_point
    history = [x]
    for _ in range(max_iter):
        grad = grad_func(x)
        x = x - learning_rate * grad
        history.append(x)
    return np.array(history)

# Решение методом BFGS с использованием scipy
def bfgs_optimizer(start_point, max_iter=1000):
    result = minimize(rosenbrock, start_point, jac=rosenbrock_grad, method='BFGS', options={'maxiter': max_iter})
    return result.x, result.nit  # Возвращаем решение и количество итераций

# Начальная точка
start_point = np.array([1.5, 2.0])

# Градиентный спуск
history_gd = gradient_descent(rosenbrock_grad, start_point)

# BFGS
solution_bfgs, n_iter_bfgs = bfgs_optimizer(start_point)

# Выводим результаты
print("Градиентный спуск:")
print("Точка решения после последней итерации:", history_gd[-1])
print("Количество итераций:", len(history_gd) - 1)

print("\nBFGS:")
print("Точка решения:", solution_bfgs)
print("Количество итераций:", n_iter_bfgs)
