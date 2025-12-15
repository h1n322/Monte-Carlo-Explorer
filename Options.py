import numpy as np
import math


class OptionEngine:
    @staticmethod
    def kernel(n, S0, K, r, sigma, T):
        """
        Оцінка вартості Європейського Колл-опціону.
        Використовує геометричний броунівський рух для фінальної ціни.
        """
        # Генерація випадкової компоненти Z ~ N(0,1)
        Z = np.random.randn(n)

        # Формула ціни на момент T (Geometric Brownian Motion)
        # ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
        drift = (r - 0.5 * sigma ** 2) * T
        diffusion = sigma * math.sqrt(T) * Z
        ST = S0 * np.exp(drift + diffusion)

        # Виплата (Payoff) = max(ST - K, 0)
        # Для Call опціону ми отримуємо прибуток, якщо ціна вища за страйк
        payoff = np.maximum(ST - K, 0.0)

        # Повертаємо суму виплат (дисконтування буде в контролері)
        # Третім аргументом повертаємо ST для побудови гістограми цін
        return (np.sum(payoff), n, ST)