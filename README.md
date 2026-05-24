# BigData_HW4 — Рекомендательные системы и Spark MLlib

Решение домашнего задания №4 по курсу Big Data на основе [BigData_HW4.md](https://github.com/SergUSProject/IntelligentSystemsAndTechnologies/blob/main/HomeWork/spark/docs/BigData_HW4.md).

## Вариант

Фамилия — Николаев.

```
alp = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
w   = [1,42,21,21,34,6,44,26,18,44,38,26,14,43,4,49,45,
       7,42,29,4,9,36,34,31,29,5,30,4,19,28,25,33]
variant = sum(dict(zip(alp, w))[c] for c in 'николаев') % 40 + 1   # = 6
task1   = variant % 3 + 1   # = 1
task2   = variant % 2 + 1   # = 1
```

- **Задание 1, вариант 1:** жанры Animation, Romance, Documentary.
- **Задание 2, вариант 1:** коллаборативная фильтрация по схожести пользователей (user-based).
- Задание 3 не выполняется.

## Структура

```
BigData_HW4/
├── README.md                  — этот файл
├── solution.py                — итоговый PySpark-скрипт (запуск одной командой)
├── run_log.txt                — лог последнего прогона
├── data/ml-latest-small/      — датасет MovieLens (small)
└── notebooks/BigData_HW4.ipynb — Jupyter-ноутбук с теми же шагами
```

## Данные

Используется [MovieLens ml-latest-small](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) — около 100 836 рейтингов, 9 742 фильма, 610 пользователей. Согласно условию, сначала прогон на small, при необходимости — на полном `ml-latest`.

Скачать данные:

```powershell
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip" -OutFile "data/ml-latest-small.zip"
Expand-Archive -Path "data/ml-latest-small.zip" -DestinationPath "data" -Force
```

## Запуск

Требуется Python 3.10+, Java 17+, PySpark 3.5.

```powershell
pip install pyspark==3.5.1 jupyter
python -X utf8 solution.py
```

Под Jupyter:

```powershell
jupyter lab
# открыть notebooks/BigData_HW4.ipynb
```

## Результаты (small dataset)

### Задание 1

Полный вывод в `run_log.txt`. Кратко:
- Разнесли строку `genres` через `explode(split('|'))`, посчитали `countDistinct(movieId)` по жанрам — лидер Drama (4 361), затем Comedy (3 756), Thriller (1 894). По варианту: Romance — 1 596, Animation — 611, Documentary — 440.
- Для каждого из жанров варианта построили четыре топ-10: по числу рейтингов (max и min при `>10`) и по среднему рейтингу (max и min при `>10`).

### Задание 2

| Метрика                                | Значение |
| -------------------------------------- | -------- |
| Среднее значение рейтинга в `train_init` | 3.5039   |
| RMSE baseline (предсказываем средний)   | 1.0504   |
| RMSE user-based CF (топ-30 соседей)     | 0.9885   |

User-based CF улучшает RMSE примерно на 5.9 % по сравнению с baseline.

## Метод user-based CF (кратко)

1. Центрируем рейтинги: `r̃(u, i) = r(u, i) − mean(u)`.
2. Считаем схожесть как косинус по центрированным рейтингам по общим фильмам — это эквивалент Pearson correlation. Отсекаем пары с менее чем 3 общими фильмами.
3. Симметризуем матрицу схожести, отбираем топ-30 соседей у каждого пользователя.
4. Прогноз: `r̂(u, i) = mean(u) + Σ sim(u, v) · r̃(v, i) / Σ |sim(u, v)|`. Где у пользователя нет соседей по фильму — подставляем `mean(u)` или общий средний.
5. Считаем RMSE на `test`.
