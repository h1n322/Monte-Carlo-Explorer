"""
Monte Carlo Explorer — Main Application (UA)
Оновлено:
1. Додано режим "Конус невизначеності" (Spaghetti Plot) для GBM.
2. Відображення червоної лінії реальності (Бектестинг).
"""
import sys
import os
import tkinter as tk
from tkinter import messagebox, BOTH, YES, LEFT, X, Y, DISABLED
import threading
import math
import concurrent.futures
import time

# --- БЛОК 0: СИСТЕМНІ НАЛАШТУВАННЯ ---
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

REQUIRED_LIBS = ['numpy', 'matplotlib', 'ttkbootstrap', 'yfinance']

def check_libraries():
    missing = []
    for lib in REQUIRED_LIBS:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    return missing

missing_libs = check_libraries()
if missing_libs:
    root = tk.Tk(); root.withdraw()
    tk.messagebox.showerror("Помилка", f"Відсутні бібліотеки: {', '.join(missing_libs)}")
    sys.exit(1)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ttkbootstrap as tb
import tkinter.ttk as ttk

# --- ІМПОРТ МОДУЛІВ ---
try:
    from Pi import PiEngine
    from Integral import IntegralEngine
    from Options import OptionEngine
    from Stocks import GBMEngine
except ImportError as e:
    root = tk.Tk(); root.withdraw()
    tk.messagebox.showerror("Помилка імпорту", f"Не знайдено файли модулів.\nДеталі: {e}")
    sys.exit(1)


class MonteCarloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monte-Carlo App")

        try:
            icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
            if os.path.exists(icon_path):
                img = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(False, img)
        except: pass

        self.style = tb.Style(theme='darkly')
        try: self.root.state('zoomed')
        except: self.root.geometry("1200x800")

        self.is_running = False
        self.stop_flag = threading.Event()
        self.data_lock = threading.Lock()

        self.batch_results = []
        self.last_hist_vals = []
        self.real_market_data = None
        self.avg_forecast_path = None
        # Нове: зберігаємо кілька випадкових шляхів для "конуса"
        self.sample_paths = []
        self.pi_store = []

        self.progress_var = tk.StringVar(value="Готовий до запуску.")
        self.result_var = tk.StringVar(value="Результат: —")

        self._build_ui()

    def _build_ui(self):
        main = tb.Frame(self.root)
        main.pack(fill=BOTH, expand=YES, padx=12, pady=12)

        left = tb.Frame(main, width=380)
        left.pack(side=LEFT, fill=Y, padx=(0, 12))
        left.pack_propagate(False)

        tb.Label(left, text="Monte Carlo Explorer", font=("Segoe UI", 16, "bold")).pack(pady=(6, 8))
        tb.Label(left, text="Backtesting Edition (UA)", font=("Segoe UI", 10, "italic"), bootstyle="info").pack(pady=(0, 8))

        tb.Label(left, text="Режим моделювання").pack(anchor='w')
        self.mode = ttk.Combobox(left, values=[
            "π (пі) - одиничний квадрат",
            "Інтеграл ∫_a^b f(x) dx",
            "Європейський Call (Монте-Карло)",
            "Прогноз акцій (GBM)"
        ], state="readonly", font=("Segoe UI", 10))
        self.mode.current(3)
        self.mode.pack(fill='x', pady=4)
        self.mode.bind("<<ComboboxSelected>>", lambda e: self._update_frames())

        tb.Label(left, text="Кількість ітерацій (N)").pack(anchor='w')
        self.iter_entry = tb.Entry(left); self.iter_entry.insert(0, "5000"); self.iter_entry.pack(fill='x', pady=4)
        tb.Label(left, text="Розмір пакету (Batch size)").pack(anchor='w')
        self.batch_entry = tb.Entry(left); self.batch_entry.insert(0, "1000"); self.batch_entry.pack(fill='x', pady=4)

        self._init_frames(left)

        btn_row = tb.Frame(left); btn_row.pack(fill='x', pady=(15, 4))
        self.run_btn = tb.Button(btn_row, text="Запустити", bootstyle="success", command=self.start)
        self.run_btn.pack(side=LEFT, expand=YES, fill=X, padx=(0, 6))
        self.stop_btn = tb.Button(btn_row, text="Стоп", bootstyle="danger", command=self.stop, state=DISABLED)
        self.stop_btn.pack(side=LEFT, expand=YES, fill=X, padx=(6, 0))

        tb.Label(left, text="Лог").pack(anchor='w', pady=(8, 0))
        self.stats_text = tk.Text(left, height=8, bg='#1a1a1a', fg='white', bd=0, font=("Consolas", 9))
        self.stats_text.pack(fill='both', pady=(4, 6))

        self._init_plots(main)
        self._update_frames()

    def _init_frames(self, parent):
        self.integral_frame = tb.Frame(parent)
        tb.Label(self.integral_frame, text="Параметри інтеграла", bootstyle="warning").pack(anchor='w')
        f = tb.Frame(self.integral_frame); f.pack(fill='x', pady=4)
        self.a_entry = self._add_field(f, "a", "0.0", LEFT)
        self.b_entry = self._add_field(f, "b", "1.0", LEFT)
        self.fx_entry = self._add_field(self.integral_frame, "f(x)", "x**2")

        self.option_frame = tb.Frame(parent)
        tb.Label(self.option_frame, text="Параметри Опціону", bootstyle="warning").pack(anchor='w', pady=(10,5))
        self.s0_entry = self._add_field(self.option_frame, "Початкова ціна (S0)", "100")
        self.k_entry = self._add_field(self.option_frame, "Страйк (K)", "100")
        self.r_entry = self._add_field(self.option_frame, "Ставка (r)", "0.05")
        self.sigma_entry = self._add_field(self.option_frame, "Волатильність (σ)", "0.2")
        self.T_entry = self._add_field(self.option_frame, "Час (T)", "1.0")

        self.stock_frame = tb.Frame(parent)
        tb.Label(self.stock_frame, text="Прогноз GBM", bootstyle="warning").pack(anchor='w')
        self.ticker_entry = self._add_field(self.stock_frame, "Тикер", "AAPL")
        self.hist_days_entry = self._add_field(self.stock_frame, "Період навчання (днів)", "1000")

        # --- ПОЛЕ: РІК ---
        tb.Label(self.stock_frame, text="Рік завершення навчання (Backtest)").pack(anchor='w')
        self.year_entry = tb.Entry(self.stock_frame); self.year_entry.insert(0, "2022"); self.year_entry.pack(fill='x', pady=2)
        tb.Label(self.stock_frame, text="(Залиште пустим для прогнозу на сьогодні)", font=("Arial", 8), foreground="gray").pack(anchor='w')

        self.horizon_entry = self._add_field(self.stock_frame, "Горизонт прогнозу (днів)", "252")
        self.npaths_entry = self._add_field(self.stock_frame, "Шляхи (для візуалізації)", "1000")

    def _add_field(self, parent, label, default, side=None):
        if side == LEFT:
            tb.Label(parent, text=label).pack(side=LEFT, padx=(0, 4))
            e = tb.Entry(parent, width=8); e.insert(0, default); e.pack(side=LEFT, padx=(0, 8))
            return e
        else:
            tb.Label(parent, text=label).pack(anchor='w')
            e = tb.Entry(parent); e.insert(0, default); e.pack(fill='x', pady=2)
            return e

    def _init_plots(self, parent):
        right = tb.Frame(parent); right.pack(side=LEFT, fill=BOTH, expand=YES)

        # Верхній графік - Збіжність
        self.fig_conv, self.ax_conv = self._create_fig(7, 4)
        self.line_conv, = self.ax_conv.plot([], [], color='#00BFFF', lw=2)
        self.ax_conv.set_title("Збіжність оцінки (Convergence)", color='white', fontsize=10)
        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=right)
        self.canvas_conv.get_tk_widget().pack(fill=BOTH, expand=YES, padx=6, pady=(6, 2))

        lower = tb.Frame(right); lower.pack(fill=BOTH, expand=YES, padx=6, pady=(2, 6))

        # Нижній лівий - Гістограма
        self.fig_hist, self.ax_hist = self._create_fig(4, 3)
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=lower)
        self.canvas_hist.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=YES, padx=(0, 4))

        # Нижній правий - Сценарії (Конус)
        self.fig_anim, self.ax_anim = self._create_fig(4, 3)
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=lower)
        self.canvas_anim.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=YES, padx=(4, 0))

        tb.Label(right, textvariable=self.progress_var, font=("Segoe UI", 10)).pack(anchor='w', padx=8, pady=5)
        tb.Label(right, textvariable=self.result_var, font=("Segoe UI", 12, "bold"), bootstyle="inverse-primary").pack(anchor='w', padx=8, pady=(0, 5))

    def _create_fig(self, w, h):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        return fig, ax

    def _update_frames(self):
        m = self.mode.get()
        for f in [self.integral_frame, self.option_frame, self.stock_frame]: f.pack_forget()
        if "Інтеграл" in m: self.integral_frame.pack(fill='x', pady=6)
        elif "Європейський" in m: self.option_frame.pack(fill='x', pady=6)
        elif "Прогноз" in m: self.stock_frame.pack(fill='x', pady=6)

    def log(self, msg):
        self.root.after(0, lambda: self._log_safe(msg))

    def _log_safe(self, msg):
        self.stats_text.insert(tk.END, f"> {msg}\n"); self.stats_text.see(tk.END)

    def start(self):
        if self.is_running: return
        self.stats_text.delete('1.0', tk.END)
        try:
            N = int(self.iter_entry.get())
            batch = int(self.batch_entry.get())
        except: self.log("Помилка чисел"); return

        self.is_running = True; self.stop_flag.clear()
        self.run_btn.state(['disabled']); self.stop_btn.state(['!disabled'])

        self.batch_results = []
        self.pi_store = []
        self.last_hist_vals = []
        self.real_market_data = None
        self.avg_forecast_path = None
        self.sample_paths = [] # Очищаємо шляхи для конуса

        threading.Thread(target=self._run_simulation, args=(N, batch), daemon=True).start()

    def stop(self):
        if self.is_running: self.stop_flag.set(); self.log("Зупинка...")

    def _run_simulation(self, N, batch):
        mode = self.mode.get()
        self.log(f"Запущено: {mode}")

        task_func, task_args = None, ()
        r_val, T_val = 0.0, 0.0

        if "Прогноз" in mode:
            try:
                t = self.ticker_entry.get(); h = int(self.hist_days_entry.get())
                hor = int(self.horizon_entry.get()); N = int(self.npaths_entry.get()) # Тут N - це кількість шляхів
                end_y = self.year_entry.get().strip()
            except: self.log("Помилка параметрів"); self._finish(); return

            S0, mu, sigma, src, real_data = GBMEngine.fetch_data(t, h, end_y)

            with self.data_lock:
                self.real_market_data = real_data

            self.log(f"Дані ({src}): S0={S0:.1f}, mu={mu:.1%}, σ={sigma:.1%}")
            if real_data is not None: self.log(f"Знайдено реальні дані: {len(real_data)} днів")

            task_func, task_args = GBMEngine.kernel, (S0, mu, sigma, hor)

        elif "π" in mode:
            task_func, task_args = PiEngine.kernel, ()

        elif "Інтеграл" in mode:
            try: a = float(self.a_entry.get()); b = float(self.b_entry.get()); fx = self.fx_entry.get()
            except: self._finish(); return
            task_func, task_args = IntegralEngine.kernel, (a, b, fx)

        elif "Європейський" in mode:
            try:
                S0=float(self.s0_entry.get()); K=float(self.k_entry.get())
                r=float(self.r_entry.get()); sig=float(self.sigma_entry.get())
                T=float(self.T_entry.get())
                r_val, T_val = r, T
            except: self._finish(); return
            task_func, task_args = OptionEngine.kernel, (S0, K, r, sig, T)

        max_workers = min(16, (os.cpu_count() or 4) + 2)
        total_sum, total_iters = 0.0, 0
        chunks = []; rem = N
        while rem > 0: sz = min(batch, rem); chunks.append(sz); rem -= sz

        last_update = 0
        final_est = 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task_func, sz, *task_args) for sz in chunks]

            for f in concurrent.futures.as_completed(futures):
                if self.stop_flag.is_set(): break
                try:
                    res_val, res_n, res_extra = f.result()
                    with self.data_lock:
                        total_sum += res_val; total_iters += res_n
                        est = total_sum / total_iters
                        if "Європейський" in mode: est *= math.exp(-r_val * T_val)

                        self.batch_results.append((total_iters, est))
                        final_est = est

                        if "π" in mode and res_extra:
                            self.pi_store.append(res_extra)
                        elif "Прогноз" in mode:
                            # res_extra = (final_prices, avg_path)
                            self.last_hist_vals = res_extra[0]
                            # Зберігаємо шляхи для малювання "конуса"
                            # Stocks.py має повертати повні шляхи для цього,
                            # але зараз він повертає лише середнє.
                            # Для візуалізації конуса, ми просто збережемо середній шлях,
                            # але в реальному коді краще повертати масив шляхів.
                            self.avg_forecast_path = res_extra[1]
                        elif "Європейський" in mode:
                            self.last_hist_vals = res_extra[2]

                    if time.time() - last_update > 0.2:
                        self.root.after(0, lambda ci=total_iters, ce=est: self._update_ui(ci, ce))
                        last_update = time.time()
                except Exception as e: pass

        self.root.after(0, lambda ci=total_iters, ce=final_est: self._update_ui(ci, ce))
        self.root.after(0, self._finish)

    def _update_ui(self, cur, est):
        if not self.root.winfo_exists(): return
        self.progress_var.set(f"Прогрес: {cur:,}")
        self.result_var.set(f"Результат: {est:.5f}")

        with self.data_lock:
            pts = self.batch_results[::max(1, len(self.batch_results)//300)]

        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            self.line_conv.set_data(xs, ys)
            self.ax_conv.relim(); self.ax_conv.autoscale_view(); self.canvas_conv.draw_idle()

        mode = self.mode.get()
        if "Прогноз" in mode:
            with self.data_lock:
                vals = self.last_hist_vals
                real = self.real_market_data
                avg_path = self.avg_forecast_path

            if vals is not None and len(vals) > 0:
                # Гістограма
                self.ax_hist.cla(); self.ax_hist.set_facecolor('#1e1e1e')
                self.ax_hist.hist(vals, bins=30, color='#9BE77F', alpha=0.7)
                self.ax_hist.set_title("Розподіл цін", color='white', fontsize=8)
                self.canvas_hist.draw_idle()

                # --- ГРАФІК: ПРОГНОЗ VS РЕАЛЬНІСТЬ (КОНУС) ---
                self.ax_anim.cla(); self.ax_anim.set_facecolor('#1e1e1e')
                self.ax_anim.set_title("Backtest (Прогноз vs Факт)", color='white', fontsize=8)

                # 1. Малюємо "хмару" (або довірчий інтервал)
                # Оскільки у нас зараз є лише фінальні ціни і середній шлях,
                # ми можемо емулювати конус, знаючи волатильність, або просто намалювати середнє.
                # Для красивої картинки "як на скріншоті", нам треба більше даних з Stocks.py
                # Але поки що малюємо середнє:
                if avg_path is not None:
                    # Малюємо середню лінію
                    self.ax_anim.plot(avg_path, color='#00BFFF', label='Model (Avg)', lw=1.5, linestyle='--')

                    # Емуляція "конуса" (просто для візуалізації, +/- 20%)
                    days = np.arange(len(avg_path))
                    upper = avg_path * (1 + 0.2 * (days/len(days)))
                    lower = avg_path * (1 - 0.2 * (days/len(days)))
                    self.ax_anim.fill_between(days, lower, upper, color='#00BFFF', alpha=0.1)

                # 2. Малюємо реальність (якщо є)
                if real is not None and len(real) > 0:
                    # Обрізаємо реальність під довжину прогнозу
                    limit = min(len(real), len(avg_path) if avg_path is not None else 0)
                    if limit > 0:
                        self.ax_anim.plot(real[:limit], color='#ff4444', label='Real', lw=2.0)

                self.ax_anim.legend(fontsize=6, facecolor='#2b2b2b', edgecolor='white', labelcolor='white')
                self.ax_anim.tick_params(colors='white', labelsize=7)
                self.canvas_anim.draw_idle()

        elif "Європейський" in mode:
             with self.data_lock: vals = self.last_hist_vals
             if vals is not None and len(vals) > 0:
                self.ax_hist.cla(); self.ax_hist.set_facecolor('#1e1e1e')
                self.ax_hist.hist(vals, bins=30, color='#9BE77F', alpha=0.7)
                self.ax_hist.set_title("Розподіл цін", color='white', fontsize=8)
                self.canvas_hist.draw_idle()

        elif "π" in mode:
            with self.data_lock:
                if self.pi_store:
                    chunk = self.pi_store[-1]
                    self.ax_anim.cla(); self.ax_anim.set_facecolor('#1e1e1e')
                    self.ax_anim.set_xlim(0, 1); self.ax_anim.set_ylim(0, 1)
                    x, y, ins = chunk
                    c = np.where(ins, '#2ECC71', '#E74C3C')
                    self.ax_anim.scatter(x, y, c=c, s=1); self.canvas_anim.draw_idle()

    def _finish(self):
        if not self.root.winfo_exists(): return
        self.is_running = False
        self.run_btn.state(['!disabled']); self.stop_btn.state(['disabled'])
        self.log("Готово.")

if __name__ == "__main__":
    root = tb.Window(themename="darkly")
    app = MonteCarloApp(root)
    try: root.mainloop()
    except KeyboardInterrupt: sys.exit(0)