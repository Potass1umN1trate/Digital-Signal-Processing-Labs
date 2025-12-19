import argparse
import random
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Утилиты
# -----------------------------
def to_gray_float(img_bgr: np.ndarray) -> np.ndarray:
    """BGR/Gray -> float32 grayscale in [0..1]."""
    if img_bgr.ndim == 3:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = img_bgr
    g = g.astype(np.float32)
    # нормировка для стабильности (не обязательно, но полезно)
    if g.max() > 1.0:
        g /= 255.0
    return g


def show_image(ax, img, title: str, cmap=None):
    ax.set_title(title)
    ax.axis("off")
    ax.imshow(img, cmap=cmap)


def log_vis(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Лог-визуализация корреляции (иначе всё будет 'яркая точка + тьма')."""
    x = np.abs(x)
    x = np.log(x + eps)
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return x


# -----------------------------
# FFT-корреляция
# -----------------------------
def cross_correlation_fft(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Взаимная корреляция через FFT:
    corr = ifft2( FFT(image) * conj(FFT(template_padded)) )

    ВАЖНО: template нужно допаддить до размера image.
    Это не "сумма попиксельно" в лоб, но математически то же самое, просто быстрее.
    """
    H, W = image.shape
    h, w = template.shape

    # допаддили шаблон до размера image
    tpl = np.zeros((H, W), dtype=np.float32)
    tpl[:h, :w] = template

    F_img = np.fft.fft2(image)
    F_tpl = np.fft.fft2(tpl)
    corr = np.fft.ifft2(F_img * np.conj(F_tpl)).real

    # corr получилась "циклическая" (как свёртка по модулю).
    # Чтобы максимум соответствовал обычному расположению, сдвигаем.
    corr = np.fft.fftshift(corr)
    return corr


def autocorrelation_fft(image: np.ndarray) -> np.ndarray:
    """
    Автокорреляция: ifft2(FFT(img)*conj(FFT(img)))
    Перед этим убираем среднее, иначе доминирует DC-компонента.
    """
    x = image.astype(np.float32)
    x = x - x.mean()
    F = np.fft.fft2(x)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    return ac


# -----------------------------
# 4а: поиск фрагмента
# -----------------------------
@dataclass
class PatchSpec:
    patch_w: int
    patch_h: int
    x: int
    y: int


def random_patch(gray: np.ndarray, patch_w: int, patch_h: int, seed: int | None = None) -> PatchSpec:
    if seed is not None:
        random.seed(seed)

    H, W = gray.shape
    if patch_w >= W or patch_h >= H:
        raise ValueError("Патч слишком большой для изображения. Уменьши --patch-size.")

    x = random.randint(0, W - patch_w - 1)
    y = random.randint(0, H - patch_h - 1)
    return PatchSpec(patch_w=patch_w, patch_h=patch_h, x=x, y=y)


def find_patch_by_corr(corr: np.ndarray, image_shape: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[int, int]:
    """
    Находим максимум корреляции и переводим его в координаты (x,y) верхнего левого угла патча.

    После fftshift центр corr = (H/2, W/2).
    Нужно аккуратно преобразовать.
    """
    H, W = image_shape
    ph, pw = patch_shape

    # максимум
    max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)

    # координаты относительно центра
    cy, cx = H // 2, W // 2
    dy = max_y - cy
    dx = max_x - cx

    # В циклической корреляции сдвиг (dy,dx) означает, что template "лучше всего" совпал,
    # если его верхний левый угол находится в (dy,dx) (с учетом wrap).
    # Приведём к нормальным координатам 0..W-1
    x0 = dx % W
    y0 = dy % H

    # Нюанс: из-за pad в (0,0) и fftshift иногда смещение “съезжает” на размер патча.
    # Чтобы стабильно получить верхний левый угол, компенсируем размер патча:
    x0 = (x0) % W
    y0 = (y0) % H

    # Убедимся, что прямоугольник влезает без wrap (иначе сдвинем внутрь)
    x0 = min(max(0, x0), W - pw)
    y0 = min(max(0, y0), H - ph)
    return x0, y0


def task_4a(image_path: str, patch_size: int, seed: int | None):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не смог прочитать файл: {image_path}")

    gray = to_gray_float(img_bgr)
    H, W = gray.shape

    # Патч делаем квадратным (по заданию можно "случайный фрагмент")
    ps = min(patch_size, W // 2, H // 2)
    spec = random_patch(gray, ps, ps, seed=seed)
    patch = gray[spec.y:spec.y + spec.patch_h, spec.x:spec.x + spec.patch_w]

    corr = cross_correlation_fft(gray, patch)
    corr_vis = log_vis(corr)

    found_x, found_y = find_patch_by_corr(corr, (H, W), (spec.patch_h, spec.patch_w))

    # рисуем прямоугольник на исходнике
    vis = img_bgr.copy()
    cv2.rectangle(vis, (found_x, found_y), (found_x + spec.patch_w, found_y + spec.patch_h), (0, 255, 0), 2)

    # для контроля: где реально был патч
    cv2.rectangle(vis, (spec.x, spec.y), (spec.x + spec.patch_w, spec.y + spec.patch_h), (255, 0, 0), 2)

    # показываем "интерфейс" из 3 окон
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    show_image(axes[0], cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), "Исходное изображение (зелёный=найдено, синий=истина)")
    show_image(axes[1], patch, f"Случайный фрагмент {ps}x{ps}", cmap="gray")
    show_image(axes[2], corr_vis, "Корреляционная функция (log-норм)", cmap="gray")
    plt.tight_layout()
    plt.show()

    print("=== 4а: результат ===")
    print(f"Истинный патч:  x={spec.x}, y={spec.y}")
    print(f"Найденный патч: x={found_x}, y={found_y}")
    print("Если зелёный почти совпал с синим — всё ок. Если нет — значит ты сломал математику/координаты.")


# -----------------------------
# 4б: автокорреляция + пики повторов
# -----------------------------
def pick_autocorr_peaks(ac: np.ndarray, top_k: int = 8, exclude_radius: int = 20) -> list[tuple[int, int, float]]:
    """
    Ищем яркие пики, кроме центрального (нулевого сдвига).
    Это даст вектора периодичности/повторов.
    """
    H, W = ac.shape
    cy, cx = H // 2, W // 2

    ac_abs = np.abs(ac).copy()

    # вырежем область вокруг центра, чтобы не выбрать "тривиальный" пик
    y0, y1 = max(0, cy - exclude_radius), min(H, cy + exclude_radius + 1)
    x0, x1 = max(0, cx - exclude_radius), min(W, cx + exclude_radius + 1)
    ac_abs[y0:y1, x0:x1] = 0.0

    peaks = []
    tmp = ac_abs.copy()
    for _ in range(top_k):
        py, px = np.unravel_index(np.argmax(tmp), tmp.shape)
        val = tmp[py, px]
        if val <= 0:
            break
        peaks.append((py, px, float(val)))

        # подавим локальную область вокруг найденного пика (чтобы не собирать один и тот же)
        sup = 10
        yy0, yy1 = max(0, py - sup), min(H, py + sup + 1)
        xx0, xx1 = max(0, px - sup), min(W, px + sup + 1)
        tmp[yy0:yy1, xx0:xx1] = 0.0

    return peaks


def task_4b(image_path: str, top_k: int, exclude_radius: int):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не смог прочитать файл: {image_path}")

    gray = to_gray_float(img_bgr)
    H, W = gray.shape

    ac = autocorrelation_fft(gray)
    ac_vis = log_vis(ac)

    peaks = pick_autocorr_peaks(ac, top_k=top_k, exclude_radius=exclude_radius)

    # визуально отметим пики на карте автокорреляции
    ac_mark = (ac_vis * 255).astype(np.uint8)
    ac_mark = cv2.cvtColor(ac_mark, cv2.COLOR_GRAY2BGR)
    cy, cx = H // 2, W // 2
    cv2.circle(ac_mark, (cx, cy), 6, (0, 255, 0), 2)

    for (py, px, _) in peaks:
        cv2.circle(ac_mark, (px, py), 6, (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    show_image(axes[0], cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), "Исходное изображение")
    show_image(axes[1], cv2.cvtColor(ac_mark, cv2.COLOR_BGR2RGB), "Автокорреляция (log-норм) + пики")
    plt.tight_layout()
    plt.show()

    print("=== 4б: выраженные повторяющиеся смещения (вектора) ===")
    print("Вектор (dx, dy) — это на сколько пикселей 'узор' повторяется.")
    for i, (py, px, val) in enumerate(peaks, 1):
        dy = py - cy
        dx = px - cx
        print(f"{i}) dx={dx:>5}, dy={dy:>5}  (сила пика ~ {val:.6f})")

    print("\nЕсли пиков нет или они хаотичные — либо картинка без повторов, либо ты взял фото, где повторяемость не выражена.")
    print("Для красивой ЛР4 бери: решётки, кирпич, плитку, ткань, фасады с окнами, текстуры.")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ЛР4: корреляция изображений (взаимная + автокорреляция)")
    parser.add_argument("--mode", choices=["cross", "auto"], required=True, help="cross=4а, auto=4б")
    parser.add_argument("--image", required=True, help="Путь к изображению")
    parser.add_argument("--patch-size", type=int, default=128, help="Размер патча для 4а (квадрат)")
    parser.add_argument("--seed", type=int, default=None, help="Seed для случайного патча (чтобы повторялось)")
    parser.add_argument("--top-k", type=int, default=8, help="Сколько пиков искать в 4б")
    parser.add_argument("--exclude-radius", type=int, default=20, help="Радиус вокруг центра автокорреляции, который игнорируем")

    args = parser.parse_args()

    if args.mode == "cross":
        task_4a(args.image, args.patch_size, args.seed)
    else:
        task_4b(args.image, args.top_k, args.exclude_radius)


if __name__ == "__main__":
    main()
