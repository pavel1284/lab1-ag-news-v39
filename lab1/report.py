"""Russian reproducible Markdown report with numbers sourced from EDA CSV."""
from datetime import date
from pathlib import Path

from lab1.config import Lab1Config
from lab1.hashes import sha256


def write_report(config: Lab1Config, r: dict) -> None:
    out = Path(config.report_dir)
    split = config.input_dir
    def classes(items):
        return "; ".join(f"{name}: **{n:,}**" for name, n in items.items())
    report = f"""# Лабораторная работа №1 — вариант 39. AG News: темы

**Дата:** {date.today():%d.%m.%Y}
**Автор:** заполнить ФИО студента
**Вариант:** 39 (текстовый профиль P3)
**Коммит кода:** `{r['git_commit']}`
**MLflow run ID:** `{r['run_id']}`
**Статус:** EDA выполнен для данных из `{split}`.

## 1. Цель

Развернуть Git-репозиторий с контролем качества коммитов и выполнить разведочный анализ
новостных текстов AG News с акцентом на распределение тем, длины текстов и словарь.

## 2. Паспорт данных

- **Источник:** https://huggingface.co/datasets/fancyzhx/ag_news (`{config.dataset_id}`).
- **Лицензия набора:** на странице Hugging Face указано `unknown`; перед внешним
  распространением текстов необходимо уточнить права оригинальных источников.
- **Состав:** поля `text` (строковый текст новости), `label` (целое 0—3).
- **Классы:** 0 World, 1 Sports, 2 Business, 3 Sci/Tech.
- **TRAIN:** {r['n_train']:,} статей; **TEST:** {r['n_test']:,} статей.
- **Хеш TRAIN (SHA-256):** `{sha256(split / 'train.csv')}`.
- **Хеш TEST (SHA-256):** `{sha256(split / 'test.csv')}`.
- **Манифест:** `reports/LAB1/hash_manifest.json`.
- **План разбиения для последующих работ:** исходный `test` не участвует в подборе
  параметров. На этапе ЛР2 из `train` выделяется воспроизводимая стратифицированная
  валидационная часть (например, 20% / seed={config.seed}); не смешивать с `test`.

## 3. Git и pre-commit

Настроены `.gitignore`, `.gitattributes`, шаблон pull request, локальный скрипт
проверок, `.pre-commit-config.yaml`, тесты и workflow CI. Блокируются Python-файлы
с ошибкой синтаксиса или стиля, строки с секретами и файлы размером свыше
{config.max_stage_bytes:,} байт. Репозиторий содержит `main` и `develop`.

Защита `main` на сервере GitHub настраивается владельцем репозитория: запретить прямые
push, потребовать pull request, минимум одно approval, обязательный успешный CI.
Локальный Git не может включить серверные Branch Protection Rules.

**Журнал негативных проверок:** `reports/LAB1/precommit_block.log`.

## 4. Разведочный анализ данных

**Распределение классов, TRAIN:** {classes(r['train_count_per_class'])}.

**Распределение классов, TEST:** {classes(r['test_count_per_class'])}.

**Структура и качество:** пустые тексты TRAIN = {r['empty_train']}, TEST = {r['empty_test']};
повторяющиеся тексты TRAIN = {r['train_duplicates']}, TEST = {r['test_duplicates']};
совпадений текстов между TRAIN и TEST = {r['train_test_overlap']}.

**Тексты и словарь:** медианная длина TRAIN = {r['median_words']:.1f} слова,
90-й процентиль = {r['p90_words']:.1f}; размер словаря (токены длиной ≥
{config.min_token_length}) = {r['vocab_size']:,}; доля токенов, встреченных один раз
(hapax), = {r['hapax_share_pct']:.2f}%.

**Сходство тематической лексики:** максимальное Жаккарово сходство двух тем по
топ-100 тематических слов — {r['max_overlap_topics']['jaccard']:.3f}
({r['max_overlap_topics']['pair']}). Это **лексическая близость**, а не фактическая
путаница предсказаний обученной модели.

### Графики

1. ![Баланс классов](figures/class_balance.png)
2. ![Длины текстов](figures/length_by_topic.png)
3. ![Частые слова по темам](figures/top_words_by_topic.png)
4. ![Частотные корзины словаря](figures/vocabulary_frequency_buckets.png)
5. ![Лексическая близость тем](figures/topic_word_overlap.png)

## 5. Выводы (каждый с числовым обоснованием)

1. Обучающая часть содержит **{r['n_train']:,}** записей — объём позволяет оценивать
   частоты слов по темам (источник: `eda_stats.csv`, метрика `records`).
2. Медианная длина учебной новости составляет **{r['median_words']:.1f}** слова —
   длины следует учитывать при выборе параметров токенизации.
3. Словарь по выбранному правилу нормализации содержит **{r['vocab_size']:,}**
   уникальных токенов — частотные ограничения снизят разреженность TF-IDF.
4. **{r['hapax_share_pct']:.2f}%** типов слов встречаются всего один раз —
   это ограничивает пользу редких слов при обучении классической модели.
5. Лексическое сходство наиболее близких двух тем равно **{r['max_overlap_topics']['jaccard']:.3f}**
   — тематическую неоднозначность стоит проверить на ЛР2.

## 6. Воспроизводимость

- `eda_stats.csv` SHA-256: `{r['eda_csv_sha256']}` (без изменений при повторе
  на тех же входных файлах и настройках).
- Графики и CSV строятся `python -m lab1.eda`.
- Манифест проверяется `python -m lab1.hashes verify`.
- Отказ на испорченном файле описан в `reports/LAB1/hash_negative_test.log`.
- Идентификатор MLflow — `{r['run_id']}`; если указано `not_logged`, EDA запускали
  в явно отключенном режиме и необходимо повторить со включённым MLflow.

## 7. Ограничения

В исходных новостях могут встречаться HTML-сущности, редакционные дубли,
различающаяся длина текста и общие слова между темами. Наличие лексического
пересечения ещё не означает ошибки классификации: для оценки качества
потребуются модели и метрики следующих лабораторных.

## 8. Источники

- AG News dataset card: https://huggingface.co/datasets/fancyzhx/ag_news
- Zhang, Zhao, LeCun (2015), *Character-level Convolutional Networks for Text Classification*.
- Исходное задание: `labs_new.docx`, раздел «ЛР1» и вариант 39.

**Источник всех рассчитанных чисел:** `reports/LAB1/eda_stats.csv`, создаваемый
`lab1/eda.py`; графики строятся тем же скриптом. Временная заглушка не используется.
"""
    (out / "lab1_report.md").write_text(report, encoding="utf-8")
