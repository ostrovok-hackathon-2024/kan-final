import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import logging
import time
import os
from functools import wraps
from ratelimit import limits, RateLimitException
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

# Настройка безопасного логирования
logging.basicConfig(filename='secure_log.log', level=logging.INFO)

plt.rcParams["figure.figsize"] = (5,5)

# Настройка лимитов запросов (Rate Limiting) для защиты от DDoS и спам-атак
REQUESTS = 10  # Максимальное количество запросов
SECONDS = 60   # Время в секундах

# Декоратор для защиты от DDoS-атак, ограничение запросов
@limits(calls=REQUESTS, period=SECONDS)
def handle_request(data):
    """
    Обработка запроса к данным с ограничением скорости запросов.
    Проверка входных данных.
    """
    # Проверка входных данных для предотвращения спам-атак
    if not isinstance(data, pd.DataFrame):
        logging.error("Неправильный тип данных")
        raise ValueError("Неправильный тип данных")

    # Обработка данных
    logging.info(f"Обработка данных началась {time.ctime()}")
    return data

# Обработка ошибки при превышении лимита запросов (DDoS-защита)
def handle_rate_limit_exception(e):
    logging.warning("Лимит запросов превышен. Попробуйте позже.")
    raise RateLimitException('Лимит запросов превышен', 403)

# Чтение пути к файлу .csv
file_path = input("Введите путь к .csv файлу: ")

# Загрузка данных из .csv файла
try:
    data = pd.read_csv(file_path)
    data = handle_request(data)
except RateLimitException as e:
    handle_rate_limit_exception(e)
except Exception as e:
    logging.error(f"Ошибка при чтении данных: {e}")
    raise ValueError("Ошибка при загрузке данных из файла")

print(data.head())
print(data.shape)
print(data.info())

# Проверка уникальности колонок
for column in data.columns:
    unique_values = data[column].nunique()
    print(f"{column}: {unique_values}")

# Пример обработки данных без шифрования
data_cut = pd.get_dummies(data.drop(['rate_name', 'class'], axis=1))
data_class = data[['rate_name', 'class']]
data = pd.concat([data_class, data_cut], axis=1)
print(data.head())
print(data.shape)

"""## Шаг 2. Провести исследовательский анализ данных (EDA)"""

# Описание данных
data_cut.describe()

# Построение гистограмм признаков
for col in data_cut.columns:
    pd.pivot_table(data, index='class', columns=col, values='rate_name', aggfunc='count').plot(kind='bar')
    plt.title(col)
    plt.ylabel('Количество')

# Корреляция признаков
data_cut.corr()

"""## Шаг 3. Построение модели прогнозирования оттока клиентов"""

# Отбор метрик и целевой переменной
X = data_cut
y = data['class']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Стандартизация данных
scaler = StandardScaler()
scaler.fit(X_train)
X_train_st = scaler.transform(X_train)
X_test_st = scaler.transform(X_test)

# Проверка данных перед обучением
def validate_data(X_train, y_train):
    assert X_train.shape[0] > 0, "Обучающие данные пусты"
    assert y_train.isin([0, 1]).all(), "Целевая переменная содержит некорректные значения"
    return True

validate_data(X_train_st, y_train)

# Логистическая регрессия
try:
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train_st, y_train)
    predictions = model.predict(X_test_st)
    probabilities = model.predict_proba(X_test_st)[:, 1]
except Exception as e:
    logging.error(f"Ошибка при обучении модели: {e}")
    raise ValueError("Ошибка при обучении модели")

# Модель дерева решений
try:
    tree_model = DecisionTreeClassifier(random_state=0)
    tree_model.fit(X_train_st, y_train)
    tree_predictions = tree_model.predict(X_test_st)
    tree_probabilities = tree_model.predict_proba(X_test_st)[:, 1]
except Exception as e:
    logging.error(f"Ошибка при обучении модели дерева решений: {e}")
    raise ValueError("Ошибка при обучении модели дерева решений")

"""## Шаг 4. Кластеризация клиентов"""

# Стандартизация данных для кластеризации
sc = StandardScaler()
X_sc = sc.fit_transform(X)

# Кластеризация с K-Means
linked = linkage(X_sc, method='ward')

# Функция отображения кластеров
def show_clusters_on_plot(df, x_name, y_name, cluster_name):
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=df[x_name], y=df[y_name], hue=df[cluster_name], palette='Paired')
    plt.title('{} vs {}'.format(x_name, y_name))
    plt.show()

# K-Means с 13 кластерами
km = KMeans(n_clusters=13, random_state=0)
labels = km.fit_predict(X_sc)

# Добавление меток кластеров к данным
data['cluster_km'] = labels

# Визуализация кластеров
col_pairs = list(itertools.combinations(data.drop(['cluster_km', 'rate_name'], axis=1).columns, 2))
for pair in col_pairs:
    show_clusters_on_plot(data, pair[0], pair[1], 'cluster_km')
