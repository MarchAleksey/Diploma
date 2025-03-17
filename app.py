import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from main import add_random_noise


# Функция для загрузки данных
def load_data(dataset_name):
    """
    Загружает датасет по имени.
    :param dataset_name: имя датасета ('iris', 'titanic', 'mnist')
    :return: DataFrame с данными и Series с метками
    """
    if dataset_name == "iris":
        # Загрузка датасета Iris из scikit-learn
        data = load_iris(as_frame=True)
        X = data.data  # Признаки
        y = data.target  # Метки
        return X, y

    elif dataset_name == "titanic":
        df = pd.read_csv("data/titanic.csv")

        # Обработка пропущенных значений
        df = df.dropna(
            subset=["Age", "Fare"]
        )  # Удаляем строки с пропусками в важных столбцах
        df["Embarked"] = df["Embarked"].fillna("S")  # Заполняем пропуски в Embarked

        # Кодирование категориальных признаков
        df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

        # Разделение на признаки и метки
        X = df.drop(columns=["Survived"])
        y = df["Survived"]
        return X, y

    elif dataset_name == "mnist":
        # Для MNIST используем Keras
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (_, _) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], -1)  # Преобразуем в плоский формат
        return pd.DataFrame(X_train), pd.Series(y_train)

    else:
        raise ValueError(f"Датасет '{dataset_name}' не поддерживается.")


# Проверка загрузки датасета Titanic
# X_titanic, y_titanic = load_data('titanic')
# print(X_titanic.head())
# print(y_titanic.head())

# Проверка загрузки датасета MNIST
# X_mnist, y_mnist = load_data('mnist')
# print(X_mnist.shape, y_mnist.shape)


# Главная функция приложения
def main():
    st.title("Исследование влияния шума в аннотациях на качество моделей")

    # Выбор датасета
    dataset_name = st.selectbox("Выберите датасет", ["iris", "titanic", "mnist"])
    try:
        X, y = load_data(dataset_name)
        st.write(f"Загружено {len(X)} примеров из датасета {dataset_name}.")
    except ValueError as e:
        st.error(e)
        return

    # Выбор уровня шума
    noise_level = st.slider(
        "Уровень шума", min_value=0.0, max_value=1.0, step=0.05, value=0.1
    )
    st.write(f"Текущий уровень шума: {noise_level:.2f}")

    # Добавление шума
    if st.button("Добавить шум"):
        y_noisy = add_random_noise(y, noise_level)
        st.write("Шум успешно добавлен!")

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Разделение данных на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_noisy, test_size=0.2, random_state=42
        )

        # Обучение модели
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Предсказание и оценка качества
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Точность модели на тестовых данных: {accuracy:.4f}")

        # Визуализация результатов
        st.subheader("Результаты обучения:")
        st.write(pd.DataFrame({"Метрика": ["Точность"], "Значение": [accuracy]}))


if __name__ == "__main__":
    main()
