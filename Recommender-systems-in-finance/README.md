# Recommender Systems for Financial Assets

Курсовой проект по построению рекомендательных систем для задачи сопоставления инвестор–актив на реальных финансовых данных.

## Задача

Предсказать, какие финансовые активы порекомендовать каждому инвестору, на основе истории транзакций и характеристик активов. Метрика качества — **nDCG@10**.

## Данные

Датасет **FAR-Trans** (Financial Asset Recommendation):
- 7 349 инвесторов, 250 активов
- Разреженность матрицы взаимодействий: **97.2%**
- Временной диапазон: разбивка train / valid / test по дате транзакций

## Модели

Сравнено **11 моделей** трёх классов:

| Класс | Модели |
|---|---|
| Коллаборативная фильтрация | User-kNN, Item-kNN, SVD, SVD++, NMF, ALS, BPR |
| Content-based | Content-aware LightGBM |
| Ансамбль | Hybrid Rank Fusion (User-kNN + NMF + LGBM) |
| Baseline | Random, Popularity |

Гиперпараметры оптимизировались с помощью **Optuna**.

## Результаты

| Модель | nDCG@10 (test) |
|---|---|
| **Hybrid Rank Fusion** | **0.485** |
| NMF | 0.444 |
| User-kNN | 0.416 |
| Content-aware LGBM | 0.402 |
| ALS | 0.390 |
| BPR | 0.319 |
| Popularity | ~0.25 |
| Random | ~0.01 |

Гибридная модель также показала превосходство над средней рыночной доходностью по метрике **ROI@10**.

## Стек

`Python` · `pandas` · `numpy` · `scikit-learn` · `scikit-surprise` · `implicit` · `LightGBM` · `Optuna` · `matplotlib` · `seaborn`

## Структура

```
├── FAR-Trans.zip # датасет
├── README.md
└── Recommender_systems_for_financial_assets.ipynb # полный пайплайн: EDA -> модели -> оценка
```

## Запуск

1. Скачать датасет **FAR-Trans.zip**, который лежит в списке файлов тут
2. Скачать файл `Recommender_systems_for_financial_assets.ipynb` из этого репозитория
3. Открыть [Google Colab](https://colab.research.google.com/) и загрузить `Recommender_systems_for_financial_assets.ipynb`
4. В боковой панели Colab (📁 Файлы) загрузить `FAR-Trans.zip`
5. Запускать ячейки по порядку — первая ячейка распакует архив автоматически
