# Модель функциональной связности генных кластеров

Автор: Зайцев Сергей Владиславович, НИУ ВШЭ, ФКН, 2 курс
Руководитель: Шмаков Сергей Анатольевич, НИУ ВШЭ

Проект: Методы машинного обучения для исследования геномных кластеров прокариот

Научно-учебная группа эволюционной геномики
---

## О проекте

Функционально связанные гены в бактериальных геномах, как правило, расположены рядом друг с другом. Эта модель принимает на вход **пару генов** (их координаты на геноме и таксономию организма) и предсказывает, принадлежат ли они одной функциональной системе — например, кластеру защиты, аннотированному PADLOC, или функциональной группе по базе COG.

**Вход:** два гена с координатами и таксономия организма  
**Выход:** `СВЯЗАНЫ` / `НЕ СВЯЗАНЫ` + уверенность в процентах

Модель обучена на данных из баз [COG2024](https://www.ncbi.nlm.nih.gov/research/cog/) и [PADLOC](https://padloc.otago.ac.nz/).

---

## Структура репозитория

```
model/
├── src/
│   ├── data.py           # Генерация датасета из сырых аннотаций
│   ├── train.py          # Обучение модели
│   ├── evaluate.py       # Оценка модели и построение графиков
│   └── predict.py        # Инференс для одной пары генов
├── requirements.txt      # Зависимости Python
└── README.md
```

---

## Установка

```bash
git clone <repo-url>
cd model

pip install -r requirements.txt
```

Требуется Python 3.10+.

---

## Запуск

### 1. Генерация датасета

Сырые данные (`data/raw/cogs.csv`) должны быть на вашей машине.

```bash
python src/data.py \
    --input data/raw/cogs.csv \
    --output data/processed/pairwise_cogs.csv
```

Чтобы быстро проверить пайплайн, можно обработать только часть данных:

```bash
python src/data.py --input data/raw/cogs.csv \
    --output data/processed/pairwise_cogs.csv \
    --sample_frac 0.1
```

### 2. Обучение модели

```bash
python src/train.py \
    --data data/processed/pairwise_cogs.csv \
    --model models/model_pipeline.joblib
```

Используется `HistGradientBoostingClassifier`. Разбивка на train/test происходит **по ID сборки генома** (`GroupShuffleSplit`), чтобы избежать утечки данных между геномами одного вида.

### 3. Оценка модели

```bash
python src/evaluate.py \
    --data data/processed/pairwise_cogs.csv \
    --model models/model_pipeline.joblib \
    --out_dir reports/figures
```

Сохраняются графики в `reports/figures/`:
- `confusion_matrix.png` — матрица ошибок
- `roc_curve.png` — ROC-кривая
- `pr_curve.png` — кривая Precision-Recall
- `feature_importance.png` — важность признаков (Permutation Importance)

### 4. Предсказание для одной пары генов

```bash
python src/predict.py \
    --cog1 COG0001 --start1 1000 --end1 2000 \
    --cog2 COG0002 --start2 2500 --end2 3500 \
    --phylum Proteobacteria \
    --tax_class Gammaproteobacteria
```

Пример вывода:

```
============================================
Gene 1:   COG0001  [1000 → 2000]
Gene 2:   COG0002  [2500 → 3500]
Taxonomy: Proteobacteria / Gammaproteobacteria
--------------------------------------------
Distance:          +500 bp
Genes between:       0
Length diff:       0 bp
--------------------------------------------
Verdict: ✅ RELATED
Confidence: 87.34%
============================================
```

---

## Признаки модели

| Признак | Описание |
|---|---|
| `distance` | Геномное расстояние между генами (пар оснований) |
| `genes_between` | Количество генов между парой |
| `length_diff` | Разница в длинах генов |
| `len_1`, `len_2` | Длины отдельных генов |
| `is_same_cog` | Принадлежат ли оба гена одному COG |
| `is_neighbor` | Расстояние < 100 пар оснований |
| `phylum`, `class` | Таксономические метки (кодируются TargetEncoder) |
