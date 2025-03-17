import numpy as np


def add_random_noise(y_true, noise_level=0.1):
    """
    Добавляет случайный шум в метки.
    :param y_true: истинные метки
    :param noise_level: уровень шума (процент ошибок)
    :return: испорченные метки
    """
    n = len(y_true)
    indices_to_flip = np.random.choice(n, int(noise_level * n), replace=False)
    y_noisy = y_true.copy()
    unique_labels = np.unique(y_true)

    for idx in indices_to_flip:
        other_labels = unique_labels[unique_labels != y_true[idx]]
        y_noisy[idx] = np.random.choice(other_labels)

    return y_noisy
