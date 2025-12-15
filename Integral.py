import numpy as np
import math


class IntegralEngine:
    # Безпечні змінні для eval()
    SAFE_GLOBALS = {
        '__builtins__': None,
        'np': np, 'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
        'log': np.log, 'sqrt': np.sqrt, 'pi': math.pi, 'x': 0
    }

    @staticmethod
    def safe_eval(expr, x_vals):
        """Безпечне обчислення виразу f(x) для масиву x."""
        local_vars = IntegralEngine.SAFE_GLOBALS.copy()
        local_vars['x'] = x_vals
        try:
            # Спроба векторизованого обчислення (швидко)
            return eval(expr, IntegralEngine.SAFE_GLOBALS, local_vars)
        except Exception:
            # Fallback для складних випадків (поелементно, повільніше)
            return np.array([eval(expr, IntegralEngine.SAFE_GLOBALS, {'x': v}) for v in x_vals])

    @staticmethod
    def kernel(n, a, b, fx_str):
        """
        Ядро обчислень інтегралу методом Монте-Карло.
        """
        # 1. Генеруємо випадкові X в діапазоні [a, b]
        xs = np.random.uniform(a, b, n)

        # 2. Обчислюємо значення функції Y = f(X)
        try:
            ys = IntegralEngine.safe_eval(fx_str, xs)
            s = np.sum(ys)
        except:
            return (0, n, None)  # Помилка у формулі

        # 3. Теорема про середнє: Integral = (b-a) * mean(f(x))
        # Тут ми повертаємо суму (vol * sum), ділення на N відбудеться в контролері
        vol = (b - a)
        return (vol * s, n, None)