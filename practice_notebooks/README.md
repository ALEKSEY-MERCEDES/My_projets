# Practice Notebooks — ML Course (HSE FCS)

Практические задания курса **«Машинное обучение»** на факультете компьютерных наук НИУ ВШЭ.

---

| № | Файл | Тема | Стек | Внешние данные |
|---|------|------|------|----------------|
| 01 | `practice_01_polars.ipynb` | Работа с табличными данными и визуализация | `polars`, `matplotlib` | Kaggle |
| 02 | `practice_02.ipynb` | Градиентный спуск с нуля: SGD, Momentum, Adam, SAG | `numpy`, `sklearn` | Нет |
| 03 | `practice_03_fixed.ipynb` | Feature engineering: базовая генерация признаков | `pandas`, `sklearn`, `catboost` | Kaggle |
| 03b | `practice_03_bonus.ipynb` | Feature engineering: продвинутые агрегации и признаки | `pandas`, `sklearn` | Kaggle |
| 04 | `practice_04_fixed.ipynb` | Нейронные сети: SPINN для уравнения Гельмгольца + CNN для UrbanSound8K | `torch`, `torchaudio`, `librosa` | UrbanSound8K |
| 05 | `practice_05.ipynb` | Решающие деревья с нуля: реализация, визуализация, тестирование | `numpy`, `sklearn` | Kaggle |
| 06 | `practice_06.ipynb` | Градиентный бустинг с нуля | `numpy`, `pandas`, `sklearn` | Kaggle |

---

## Запуск

Ноутбуки рассчитаны на запуск в **Kaggle Notebooks** или **Google Colab**.

- **practice_02, practice_04 (часть 1)** — внешние данные не нужны, запускаются сразу.
- **practice_01, 03, 03bonus, 05, 06** — используют датасеты с Kaggle (`alexetmercedes/datasets`). Откройте ноутбук в Kaggle и подключите датасет в разделе **Input**.
- **practice_04 (часть 2)** — требует датасет [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) (~6 ГБ), скачивается отдельно.
