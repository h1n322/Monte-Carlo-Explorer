import numpy as np


class PiEngine:
    @staticmethod
    def kernel(n, *args):
        """
        Ядро обчислень для числа Пі.
        Генерує точки в квадраті і рахує, скільки потрапило в коло.
        """
        # Генеруємо n випадкових точок
        x = np.random.rand(n)
        y = np.random.rand(n)

        # Перевірка умови кола: x^2 + y^2 <= 1
        inside = (x * x + y * y) <= 1.0

        # Підготовка даних для візуалізації (беремо не більше 2000 точок для економії пам'яті)
        vis_data = None
        if n > 0:
            limit = min(n, 2000)
            # Випадковий вибір індексів для візуалізації
            idx = np.random.choice(n, limit, replace=False)
            vis_data = (x[idx], y[idx], inside[idx])

        count = np.count_nonzero(inside)

        # Формула площі: Area = 4 * (inside / total)
        return (4.0 * count, n, vis_data)