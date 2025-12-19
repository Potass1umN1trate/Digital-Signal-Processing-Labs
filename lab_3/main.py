import cv2
import numpy as np


# ==========================
#  Вспомогательные функции
# ==========================

def load_gray_image(path: str) -> np.ndarray:
    """
    Читает изображение и переводит в оттенки серого (uint8, 0..255).
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать файл: {path}")
    return img


def save_gray_image(path: str, img: np.ndarray) -> None:
    """
    Сохраняет изображение (предварительно обрезая значения в диапазон 0..255).
    """
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_clipped)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Свёртка 2D для одноканального изображения.

    - image: 2D массив (H x W)
    - kernel: 2D массив (kh x kw)
    Границы: используем отражение (mode='edge'), чтобы не терять информацию на краях.
    """
    # Размеры изображения и ядра
    h, w = image.shape
    kh, kw = kernel.shape

    # Половины размеров ядра (для паддинга)
    pad_h = kh // 2
    pad_w = kw // 2

    # Разворачиваем ядро (классическая свёртка разворачивает ядро по X и Y)
    kernel_flipped = np.flipud(np.fliplr(kernel))

    # Паддинг изображения (отражение по краям)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Выходное изображение
    output = np.zeros_like(image, dtype=np.float64)

    # Проходим по всем пикселям исходного изображения
    for i in range(h):
        for j in range(w):
            # Вырезаем окно той же размерности, что и ядро
            region = padded[i:i + kh, j:j + kw]
            # Скалярное произведение окна и ядра
            value = np.sum(region * kernel_flipped)
            output[i, j] = value

    return output


# ==========================
#  Фильтры свёртки
# ==========================

def box_blur(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Коробочное размытие. ksize - размер окна (3, 5, 7, ...).
    Ядро заполнено единицами, нормированное на ksize*ksize.
    """
    kernel = np.ones((ksize, ksize), dtype=np.float64) / (ksize * ksize)
    return convolve2d(image, kernel)


def gaussian_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Генерирует 2D ядро Гаусса размером ksize x ksize с дисперсией sigma^2.
    ksize должен быть нечётным.
    """
    assert ksize % 2 == 1, "Размер ядра должен быть нечётным"

    # Координатная сетка от -r до r
    r = ksize // 2
    x, y = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))

    # Формула Гаусса
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Нормировка: сумма элементов ядра должна быть 1
    kernel /= np.sum(kernel)
    return kernel


def gaussian_blur(image: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Размытие по Гауссу при помощи собственной реализации ядра + свёртки.
    """
    kernel = gaussian_kernel(ksize, sigma)
    return convolve2d(image, kernel)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Медианный фильтр. Для каждого пикселя берём окно ksize x ksize
    и заменяем значение на медиану элементов окна.
    """
    assert ksize % 2 == 1, "Размер окна должен быть нечётным"

    h, w = image.shape
    r = ksize // 2

    # Паддинг для удобной обработки краёв
    padded = np.pad(image, ((r, r), (r, r)), mode='edge')
    output = np.zeros_like(image, dtype=np.float64)

    for i in range(h):
        for j in range(w):
            window = padded[i:i + ksize, j:j + ksize]
            median_value = np.median(window)
            output[i, j] = median_value

    return output


def sobel_operator(image: np.ndarray) -> np.ndarray:
    """
    Оператор Собеля.
    Возвращает карту границ (градиент по модулю).

    Используем два ядра:
    Gx = [-1 0 1; -2 0 2; -1 0 1]
    Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    """
    # Ядра Собеля по X и Y
    kernel_gx = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float64
    )

    kernel_gy = np.array(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=np.float64
    )

    # Свёртка с ядрами
    gx = convolve2d(image, kernel_gx)
    gy = convolve2d(image, kernel_gy)

    # Модуль градиента
    magnitude = np.sqrt(gx**2 + gy**2)

    # Нормировка в диапазон 0..255
    magnitude = magnitude / (magnitude.max() + 1e-8) * 255.0

    return magnitude


# ==========================
#  Пример запуска ЛР3
# ==========================

def main():
    # Путь к входному изображению
    input_path = "input.jpg"   # поменяй на своё
    img = load_gray_image(input_path)

    # 1. Коробочное размытие
    box = box_blur(img, ksize=21)
    save_gray_image("output_box_blur.png", box)

    # 2. Гауссово размытие
    gauss = gaussian_blur(img, ksize=21, sigma=5.0)
    save_gray_image("output_gaussian_blur.png", gauss)

    # 3. Медианный фильтр
    med = median_filter(img, ksize=3)
    save_gray_image("output_median.png", med)

    # 4. Оператор Собеля (карта границ)
    sobel = sobel_operator(img)
    save_gray_image("output_sobel.png", sobel)

    print("Готово: сохранены output_box_blur.png, output_gaussian_blur.png, output_median.png, output_sobel.png")


if __name__ == "__main__":
    main()
