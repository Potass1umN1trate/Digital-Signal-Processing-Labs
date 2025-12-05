import sys
import wave
import struct
import os

import numpy as np
import matplotlib.pyplot as plt


# ============================
# Чтение/Запись WAV
# ============================

def read_wavefile(filename):
    """
    Читает WAV-файл и возвращает (fs, signal_float32 в диапазоне [-1, 1]).
    Если стерео — берём первый канал.
    """
    with wave.open(filename, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Ожидается 16-битный WAV (2 байта на сэмпл)")

    data_int16 = np.frombuffer(raw, dtype=np.int16)

    if n_channels > 1:
        data_int16 = data_int16.reshape(-1, n_channels)
        data_int16 = data_int16[:, 0]

    signal = data_int16.astype(np.float32) / 32768.0
    return fs, signal

def write_wavefile(filename, signal, fs):
    """
    Записывает сигнал (float32/float64 в [-1, 1]) в 16-битный моно WAV.
    """
    # на всякий случай ограничим амплитуду, чтобы не было клиппинга
    sig_clipped = np.clip(signal, -1.0, 1.0)
    data_int16 = (sig_clipped * 32767).astype(np.int16)

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)      # моно
        wf.setsampwidth(2)      # 2 байта = 16 бит
        wf.setframerate(fs)     # частота дискретизации
        wf.writeframes(data_int16.tobytes())

# ============================
# Ручной ДПФ / ОБПФ
# ============================

def dft(x):
    """
    Прямое дискретное преобразование Фурье (ДПФ).
    На вход:  x[n] (1D-массив, время).
    На выход: X[k] (1D-массив комплексных коэффициентов).
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))  # столбец

    # Матрица из экспонент e^(-j 2π kn/N)
    M = np.exp(-2j * np.pi * k * n / N)
    X = M.dot(x)
    return X


def idft(X):
    """
    Обратное ДПФ.
    На вход:  X[k] (спектр).
    На выход: x[n] (время, вещественная часть).
    """
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))

    M = np.exp(2j * np.pi * k * n / N)
    x = M.dot(X) / N
    return x.real


# ============================
# Ручной БПФ / ОБПФ (radix-2)
# ============================

def next_pow2(n):
    """Ближайшая степень двойки >= n."""
    return 1 << (n - 1).bit_length()


def _fft_recursive(x):
    """
    Рекурсивный алгоритм БПФ Cooley–Tukey.
    Предполагаем, что длина x — степень двойки.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N == 1:
        return x.copy()

    # чётные и нечётные индексы
    X_even = _fft_recursive(x[0::2])
    X_odd = _fft_recursive(x[1::2])

    k = np.arange(N)
    factor = np.exp(-2j * np.pi * k / N)

    half = N // 2
    X = np.zeros(N, dtype=np.complex128)
    X[:half] = X_even + factor[:half] * X_odd
    X[half:] = X_even - factor[:half] * X_odd
    return X


def fft_manual(x):
    """
    БПФ:
    - приводим сигнал к комплексному виду;
    - дополняем нулями до длины — степени двойки;
    - вызываем рекурсивный _fft_recursive.

    Возвращает (X, N_orig):
      X      - спектр длины N_fft (степень двойки),
      N_orig - исходная длина сигнала.
    """
    x = np.asarray(x, dtype=np.complex128)
    N_orig = x.shape[0]
    N_fft = next_pow2(N_orig)

    if N_fft != N_orig:
        x_padded = np.zeros(N_fft, dtype=np.complex128)
        x_padded[:N_orig] = x
        x = x_padded

    X = _fft_recursive(x)
    return X, N_orig


def ifft_manual(X, N_orig=None):
    """
    Обратное БПФ через трюк:
      IFFT(X) = conj(FFT(conj(X))) / N

    X       - спектр (длина N_fft — степень двойки),
    N_orig  - исходная длина сигнала (чтобы обрезать хвост паддинга).
    """
    X = np.asarray(X, dtype=np.complex128)
    N_fft = X.shape[0]

    # FFT от комплексно-сопряжённого спектра
    X_conj = np.conjugate(X)
    x_tmp, _ = fft_manual(X_conj)  # длина уже степень двойки → паддинга не будет
    x_time = np.conjugate(x_tmp) / N_fft

    if N_orig is not None and N_orig < N_fft:
        x_time = x_time[:N_orig]

    return x_time.real


# ============================
# Вспомогательные: частотная ось
# ============================

def fft_freqs(N, fs):
    """
    Строит массив частот для FFT/DFT длины N при частоте дискретизации fs.
    Порядок такой же, как у FFT:
      [0, 1, 2, ..., N/2, ..., N-1] →
      [0, df, 2df, ..., fs/2, ..., -df]

    df = fs / N.
    """
    k = np.arange(N)
    freqs = k * fs / N
    # для индексов после N/2 считаем отрицательные частоты
    freqs[k > N // 2] -= fs
    return freqs


# ============================
# Анализ с ДПФ
# ============================

def analyze_with_dft(signal, fs, title="Сигнал", segment_N=1024):
    """
    Анализ сигнала с помощью ДПФ:
      - берём первые segment_N отсчётов,
      - считаем ДПФ и ОБПФ,
      - рисуем:
        * исходный и восстановленный во времени,
        * амплитудный спектр,
        * фазовый спектр.
    """
    x = signal[:segment_N]
    N = len(x)
    t = np.arange(N) / fs

    X = dft(x)
    x_rec = idft(X)

    # ошибки восстановления
    max_err = np.max(np.abs(x - x_rec))
    print(f"Макс. ошибка восстановления (ДПФ) для сигнала {title}: {max_err:.3e}")

    amp = np.abs(X)
    phase = np.angle(X)

    freqs = fft_freqs(N, fs)
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    amp_pos = amp[pos_mask]
    phase_pos = phase[pos_mask]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(f"Сигнал {title}: анализ с ДПФ")

    # временная область
    ax = axes[0]
    ax.plot(t, x, label="Исходный")
    ax.plot(t, x_rec, '--', label="Восстановленный")
    ax.set_ylabel("x(t)")
    ax.set_xlabel("t, с")
    ax.grid(True)
    ax.legend()

    # амплитудный спектр
    ax = axes[1]
    ax.stem(freqs_pos, amp_pos, basefmt=" ")
    ax.set_ylabel("|X|")
    ax.set_xlabel("f, Гц")
    ax.set_title("Амплитудный спектр |X[k]| (ДПФ)")
    ax.grid(True)

    # фазовый спектр
    ax = axes[2]
    ax.stem(freqs_pos, phase_pos, basefmt=" ")
    ax.set_ylabel("Фаза, рад")
    ax.set_xlabel("f, Гц")
    ax.set_title("Фазовый спектр arg(X[k]) (ДПФ)")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# ============================
# Анализ с БПФ (FFT)
# ============================

def analyze_with_fft(signal, fs, title="Сигнал", segment_N=1024):
    """
    Аналогичный анализ, но с использованием БПФ.
    Графики по виду такие же, как для ДПФ.
    """
    x = signal[:segment_N]
    N = len(x)
    t = np.arange(N) / fs

    X, N_orig = fft_manual(x)
    x_rec = ifft_manual(X, N_orig)

    max_err = np.max(np.abs(x - x_rec))
    print(f"Макс. ошибка восстановления (БПФ) для сигнала {title}: {max_err:.3e}")

    amp = np.abs(X)
    phase = np.angle(X)

    freqs = fft_freqs(len(X), fs)
    pos_mask = freqs >= 0
    freqs_pos = freqs[pos_mask]
    amp_pos = amp[pos_mask]
    phase_pos = phase[pos_mask]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(f"Сигнал {title}: анализ с БПФ")

    # временная область
    ax = axes[0]
    ax.plot(t, x, label="Исходный")
    ax.plot(t, x_rec, '--', label="Восстановленный")
    ax.set_ylabel("x(t)")
    ax.set_xlabel("t, с")
    ax.grid(True)
    ax.legend()

    # амплитудный спектр
    ax = axes[1]
    ax.stem(freqs_pos, amp_pos, basefmt=" ")
    ax.set_ylabel("|X|")
    ax.set_xlabel("f, Гц")
    ax.set_title("Амплитудный спектр |X(f)| (БПФ)")
    ax.grid(True)

    # фазовый спектр
    ax = axes[2]
    ax.stem(freqs_pos, phase_pos, basefmt=" ")
    ax.set_ylabel("Фаза, рад")
    ax.set_xlabel("f, Гц")
    ax.set_title("Фазовый спектр arg(X(f)) (БПФ)")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# ============================
# Идеальные фильтры в частотной области
# ============================

def build_ideal_filters(N_fft, fs, fc_lp=500.0, fc_hp=500.0, band=(400.0, 700.0)):
    """
    Строит идеальные прямоугольные характеристики НЧ, ВЧ и полосового фильтра.
    Возвращает (freqs, H_lp, H_hp, H_bp).
    """
    freqs = fft_freqs(N_fft, fs)
    abs_f = np.abs(freqs)

    H_lp = (abs_f <= fc_lp).astype(float)
    H_hp = (abs_f >= fc_hp).astype(float)

    f1, f2 = band
    H_bp = ((abs_f >= f1) & (abs_f <= f2)).astype(float)

    return freqs, H_lp, H_hp, H_bp


def apply_filter_in_freq(x, fs, H):
    """
    Применяет фильтр в частотной области:
      1) БПФ сигнала,
      2) умножение на H(f),
      3) ОБПФ.
    """
    X, N_orig = fft_manual(x)
    if len(H) != len(X):
        raise ValueError("Длина H и X должна совпадать")
    Y = X * H
    y = ifft_manual(Y, N_orig)
    return y


def plot_filters_separately(freqs, H_lp, H_hp, H_bp):
    """
    Рисует три частотные характеристики:
      - НЧ
      - ВЧ
      - полосовой
    Каждая на своём подграфике.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Частотные характеристики идеальных фильтров")

    # НЧ
    ax = axes[0]
    ax.plot(freqs, H_lp)
    ax.set_ylabel("H_NЧ(f)")
    ax.grid(True)

    # ВЧ
    ax = axes[1]
    ax.plot(freqs, H_hp, color='orange')
    ax.set_ylabel("H_ВЧ(f)")
    ax.grid(True)

    # Полосовой
    ax = axes[2]
    ax.plot(freqs, H_bp, color='green')
    ax.set_ylabel("H_ПФ(f)")
    ax.set_xlabel("f, Гц")
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def demo_filtering(signal, fs, title="Сигнал",
                   fc_lp=500.0, fc_hp=500.0, band=(400.0, 700.0),
                   segment_N=2048):
    """
    Демонстрация фильтрации:
      - по сегменту рисуем графики;
      - по всему сигналу фильтруем и сохраняем WAV.
    """
    # ==== 1) СЕГМЕНТ ДЛЯ ГРАФИКОВ ====
    x_seg = signal[:segment_N]
    t_seg = np.arange(len(x_seg)) / fs

    # FFT/ IFFT для сегмента (без фильтра)
    X_seg, N_orig_seg = fft_manual(x_seg)
    x_rec_seg = ifft_manual(X_seg, N_orig_seg)

    # Фильтры для сегмента
    freqs_seg, H_lp_seg, H_hp_seg, H_bp_seg = build_ideal_filters(
        len(X_seg), fs, fc_lp, fc_hp, band
    )

    # Фильтрация сегмента (для показа во времени)
    x_lp_seg = apply_filter_in_freq(x_seg, fs, H_lp_seg)
    x_hp_seg = apply_filter_in_freq(x_seg, fs, H_hp_seg)
    x_bp_seg = apply_filter_in_freq(x_seg, fs, H_bp_seg)

    # ==== ГРАФИКИ ВРЕМЯ-ДОМЕН ====
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Сигнал {title}: фильтрация в частотной области (сегмент)")

    # 1) Исходный vs восстановленный без фильтра
    ax = axes[0]
    ax.plot(t_seg, x_seg, label="Исходный")
    ax.plot(t_seg, x_rec_seg, '--', label="FFT→IFFT (без фильтра)")
    ax.set_ylabel("x(t)")
    ax.grid(True)
    ax.legend()

    # 2) НЧ
    ax = axes[1]
    ax.plot(t_seg, x_seg, color='gray', alpha=0.5, label="Исходный")
    ax.plot(t_seg, x_lp_seg, label=f"НЧ, f_c={fc_lp} Гц")
    ax.set_ylabel("НЧ")
    ax.grid(True)
    ax.legend()

    # 3) ВЧ
    ax = axes[2]
    ax.plot(t_seg, x_seg, color='gray', alpha=0.5, label="Исходный")
    ax.plot(t_seg, x_hp_seg, label=f"ВЧ, f_c={fc_hp} Гц")
    ax.set_ylabel("ВЧ")
    ax.grid(True)
    ax.legend()

    # 4) Полосовой
    f1, f2 = band
    ax = axes[3]
    ax.plot(t_seg, x_seg, color='gray', alpha=0.5, label="Исходный")
    ax.plot(t_seg, x_bp_seg, label=f"ПФ, {int(f1)}..{int(f2)} Гц")
    ax.set_ylabel("ПФ")
    ax.set_xlabel("t, с")
    ax.grid(True)
    ax.legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Частотные характеристики (можно по сегменту, форма не изменится)
    plot_filters_separately(freqs_seg, H_lp_seg, H_hp_seg, H_bp_seg)

    # ==== 2) ПОЛНЫЙ СИГНАЛ ДЛЯ ЗАПИСИ WAV ====

    # FFT/ IFFT для всего сигнала (без фильтра)
    X_full, N_orig_full = fft_manual(signal)
    x_rec_full = ifft_manual(X_full, N_orig_full)

    # Фильтры для полного сигнала
    freqs_full, H_lp_full, H_hp_full, H_bp_full = build_ideal_filters(
        len(X_full), fs, fc_lp, fc_hp, band
    )

    # Фильтрация всего сигнала
    x_lp_full = apply_filter_in_freq(signal, fs, H_lp_full)
    x_hp_full = apply_filter_in_freq(signal, fs, H_hp_full)
    x_bp_full = apply_filter_in_freq(signal, fs, H_bp_full)

    # ==== ЗАПИСЬ WAV ====
    out_dir = "out_wav"
    os.makedirs(out_dir, exist_ok=True)

    base_name = title  # имя файла без расширения

    write_wavefile(os.path.join(out_dir, f"{base_name}_fft_ifft.wav"),
                   x_rec_full, fs)
    write_wavefile(os.path.join(out_dir, f"{base_name}_lp_{int(fc_lp)}Hz.wav"),
                   x_lp_full, fs)
    write_wavefile(os.path.join(out_dir, f"{base_name}_hp_{int(fc_hp)}Hz.wav"),
                   x_hp_full, fs)
    write_wavefile(os.path.join(out_dir, f"{base_name}_bp_{int(f1)}_{int(f2)}Hz.wav"),
                   x_bp_full, fs)


# ============================
# main
# ============================

def main():
    if len(sys.argv) < 2:
        print("Использование: python3 main.py path/to/file.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"Файл {wav_path} не найден")
        sys.exit(1)

    fs, sig = read_wavefile(wav_path)
    print(f"Прочитан файл: {wav_path}")
    print(f"Частота дискретизации: {fs} Гц, длина: {len(sig)} сэмплов, длительность ~{len(sig)/fs:.2f} с")

    base = os.path.splitext(os.path.basename(wav_path))[0]

    # Анализ ДПФ
    analyze_with_dft(sig, fs, title=base, segment_N=1024)

    # Анализ БПФ
    analyze_with_fft(sig, fs, title=base, segment_N=1024)

    # Демонстрация фильтров
    demo_filtering(sig, fs, title=base, fc_lp=500.0, fc_hp=500.0, band=(400.0, 700.0),
                   segment_N=2048)

    # Показываем все фигуры разом
    plt.show()


if __name__ == "__main__":
    main()
