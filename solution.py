"""
ДЗ-4. Рекомендательные системы и Spark MLlib.
Фамилия: Николаев  ->  Задача 1: вариант 1 (Animation, Romance, Documentary)
                       Задача 2: вариант 1 (user-based collaborative filtering)
"""

import os
import sys
import math

# Принудительно UTF-8 для stdout/stderr — чтобы названия фильмов
# вроде "WALL·E (2008)" не падали с UnicodeEncodeError на cp1251.
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import DoubleType

DATA_DIR = "data/ml-latest-small"
GENRES_VARIANT_1 = ["Animation", "Romance", "Documentary"]
SEED = 42


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("BigData_HW4")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark: SparkSession):
    ratings = spark.read.csv(f"{DATA_DIR}/ratings.csv", header=True, inferSchema=True)
    movies = spark.read.csv(f"{DATA_DIR}/movies.csv", header=True, inferSchema=True)
    # Жанры в movies.csv — это строка вида "Adventure|Animation|...". Разнесём
    # по строкам, чтобы было удобно фильтровать и группировать.
    movies_exploded = movies.withColumn(
        "genre", F.explode(F.split(F.col("genres"), "\\|"))
    )
    return ratings, movies, movies_exploded


# ---------------------------------------------------------------------------
# Задание 1. Анализ датасета
# ---------------------------------------------------------------------------
def task1(ratings, movies, movies_exploded):
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 1. АНАЛИЗ ДАТАСЕТА")
    print("Вариант 1: жанры — Animation, Romance, Documentary")
    print("=" * 70)

    # 1. количество фильмов по жанрам
    genre_counts = (
        movies_exploded.groupBy("genre")
        .agg(F.countDistinct("movieId").alias("movies_cnt"))
        .orderBy(F.desc("movies_cnt"))
    )
    print("\n[1.1] Количество фильмов по жанрам:")
    genre_counts.show(50, truncate=False)

    # Подсчёт количества рейтингов и среднего рейтинга у каждого фильма
    movie_stats = (
        ratings.groupBy("movieId")
        .agg(
            F.count("rating").alias("ratings_cnt"),
            F.avg("rating").alias("avg_rating"),
        )
    )

    # Соединим: фильм + его жанр + статистика
    movie_full = (
        movies_exploded
        .join(movie_stats, on="movieId", how="left")
        .fillna({"ratings_cnt": 0, "avg_rating": 0.0})
    )

    for genre in GENRES_VARIANT_1:
        print("\n" + "-" * 70)
        print(f"ЖАНР: {genre}")
        print("-" * 70)

        gdf = movie_full.filter(F.col("genre") == genre)

        print(f"\n[1.2] Топ-10 {genre} по наибольшему числу рейтингов:")
        (gdf.orderBy(F.desc("ratings_cnt"), "title")
            .select("movieId", "title", "ratings_cnt", "avg_rating")
            .show(10, truncate=False))

        print(f"\n[1.3] Топ-10 {genre} по наименьшему числу рейтингов (>10):")
        (gdf.filter(F.col("ratings_cnt") > 10)
            .orderBy(F.asc("ratings_cnt"), "title")
            .select("movieId", "title", "ratings_cnt", "avg_rating")
            .show(10, truncate=False))

        print(f"\n[1.4] Топ-10 {genre} по наибольшему ср. рейтингу (#ratings>10):")
        (gdf.filter(F.col("ratings_cnt") > 10)
            .orderBy(F.desc("avg_rating"), F.desc("ratings_cnt"))
            .select("movieId", "title", "ratings_cnt", "avg_rating")
            .show(10, truncate=False))

        print(f"\n[1.5] Топ-10 {genre} по наименьшему ср. рейтингу (#ratings>10):")
        (gdf.filter(F.col("ratings_cnt") > 10)
            .orderBy(F.asc("avg_rating"), F.desc("ratings_cnt"))
            .select("movieId", "title", "ratings_cnt", "avg_rating")
            .show(10, truncate=False))


# ---------------------------------------------------------------------------
# Задание 2. Коллаборативная фильтрация (user-based)
# ---------------------------------------------------------------------------
def task2(ratings):
    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 2. КОЛЛАБОРАТИВНАЯ ФИЛЬТРАЦИЯ (user-based, вариант 1)")
    print("=" * 70)

    # 1) split 0.8 / 0.2 + baseline
    train_init, test = ratings.randomSplit([0.8, 0.2], seed=SEED)
    train_init.cache()
    test.cache()
    print(f"train_init: {train_init.count()} rows, test: {test.count()} rows")

    mean_rating = train_init.agg(F.avg("rating")).first()[0]
    print(f"\nСреднее значение рейтинга в train: {mean_rating:.4f}")

    rmse_baseline = (
        test.select(((F.col("rating") - F.lit(mean_rating)) ** 2).alias("se"))
        .agg(F.avg("se")).first()[0]
    )
    rmse_baseline = math.sqrt(rmse_baseline)
    print(f"RMSE (baseline = mean): {rmse_baseline:.4f}")

    # 2) user-based CF
    # Для расчёта схожести центрируем рейтинги (вычитаем среднее пользователя),
    # потом косинус по общим фильмам — это эквивалент Pearson correlation.
    user_means = (
        train_init.groupBy("userId").agg(F.avg("rating").alias("user_mean"))
    )
    train_centered = (
        train_init.join(user_means, on="userId")
        .withColumn("r_centered", F.col("rating") - F.col("user_mean"))
    )

    # Самосоединение по фильму, чтобы получить пары пользователей,
    # которые оценили один и тот же фильм.
    a = train_centered.alias("a")
    b = train_centered.alias("b")
    pairs = (
        a.join(b, on="movieId")
        .filter(F.col("a.userId") < F.col("b.userId"))
        .select(
            F.col("a.userId").alias("u"),
            F.col("b.userId").alias("v"),
            F.col("a.r_centered").alias("ru"),
            F.col("b.r_centered").alias("rv"),
        )
    )

    sim = (
        pairs.groupBy("u", "v")
        .agg(
            F.sum(F.col("ru") * F.col("rv")).alias("dot"),
            F.sum(F.col("ru") * F.col("ru")).alias("nu"),
            F.sum(F.col("rv") * F.col("rv")).alias("nv"),
            F.count("*").alias("common"),
        )
        # отсекаем редких соседей и нулевые знаменатели
        .filter((F.col("common") >= 3) & (F.col("nu") > 0) & (F.col("nv") > 0))
        .withColumn(
            "sim",
            F.col("dot") / (F.sqrt(F.col("nu")) * F.sqrt(F.col("nv"))),
        )
        .select("u", "v", "sim")
    )

    # Симметричность: дублируем пары (v, u)
    sim_full = (
        sim.select("u", "v", "sim")
        .union(sim.select(F.col("v").alias("u"), F.col("u").alias("v"), "sim"))
    )

    # Оставим топ-K соседей для каждого пользователя — иначе дорого и шумно.
    K = 30
    w = Window.partitionBy("u").orderBy(F.desc("sim"))
    top_neighbors = (
        sim_full.withColumn("rk", F.row_number().over(w))
        .filter(F.col("rk") <= K)
        .select(F.col("u").alias("userId"),
                F.col("v").alias("neighborId"),
                "sim")
    )

    # Прогноз: r̂(u, i) = mean(u) + Σ sim(u, v) * (r(v, i) - mean(v)) / Σ |sim(u, v)|
    predictions = (
        test.alias("t")
        .join(top_neighbors.alias("n"), on="userId", how="left")
        .join(
            train_centered.select(
                F.col("userId").alias("neighborId"),
                F.col("movieId"),
                F.col("r_centered"),
            ).alias("nc"),
            on=["neighborId", "movieId"],
            how="left",
        )
        .filter(F.col("r_centered").isNotNull())
        .groupBy("userId", "movieId")
        .agg(
            F.sum(F.col("sim") * F.col("r_centered")).alias("num"),
            F.sum(F.abs(F.col("sim"))).alias("den"),
        )
        .filter(F.col("den") > 0)
        .withColumn("pred_offset", F.col("num") / F.col("den"))
    )

    # Прибавляем средний рейтинг пользователя; для холодных юзеров — общий mean.
    predictions = (
        predictions.join(user_means, on="userId", how="left")
        .withColumn(
            "prediction",
            F.coalesce(F.col("user_mean"), F.lit(mean_rating))
            + F.col("pred_offset"),
        )
        # ограничим [0.5, 5.0] на всякий случай
        .withColumn("prediction", F.greatest(F.lit(0.5), F.col("prediction")))
        .withColumn("prediction", F.least(F.lit(5.0), F.col("prediction")))
        .select("userId", "movieId", "prediction")
    )

    # Полная сборка прогнозов: где CF дала ответ — берём его, иначе fallback = mean(user) или global mean.
    test_with_pred = (
        test.join(predictions, on=["userId", "movieId"], how="left")
        .join(user_means, on="userId", how="left")
        .withColumn(
            "prediction",
            F.coalesce(
                F.col("prediction"),
                F.col("user_mean"),
                F.lit(mean_rating),
            ),
        )
    )

    # 3) RMSE
    coverage = (
        test_with_pred.filter(F.col("prediction").isNotNull()).count()
        / max(test.count(), 1)
    )
    se = ((F.col("rating") - F.col("prediction")) ** 2).cast(DoubleType())
    rmse_cf = math.sqrt(
        test_with_pred.select(F.avg(se).alias("mse")).first()[0]
    )

    print(f"\nRMSE (user-based CF): {rmse_cf:.4f}")
    print(f"RMSE (baseline):      {rmse_baseline:.4f}")
    print(f"Покрытие test предсказаниями (с fallback): {coverage:.2%}")

    print("\nПримеры предсказаний:")
    test_with_pred.select("userId", "movieId", "rating", "prediction").show(10)


def main():
    spark = build_spark()
    try:
        ratings, movies, movies_exploded = load_data(spark)
        task1(ratings, movies, movies_exploded)
        task2(ratings)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
