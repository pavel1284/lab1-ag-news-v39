# Лабораторная работа №1 — вариант 39 «AG News: темы»

Выполнены: структура Git, локальные pre-commit проверки, тесты негативных
сценариев, проверка SHA-256, код разведочного анализа **полного** AG News,
машиночитаемые сводки, графики и генератор Markdown-отчёта.

## Быстрый запуск (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pre_commit install
python scripts/run_all.py
```

Для Linux/macOS: `python3 -m venv .venv`, `source .venv/bin/activate`.
Понадобится интернет для **первой** загрузки `fancyzhx/ag_news` из Hugging Face.

Отдельные шаги:

```bash
python -m lab1.download          # сохраняет полные CSV и SHA256-манифест
python -m lab1.hashes verify     # должен вывести VERDICT: MATCH
python -m lab1.eda              # EDA, CSV, 5 графиков, Markdown-отчёт, MLflow
python scripts/negative_controls.py
python scripts/repeat_eda.py
python -m pytest -q
```

MLflow ведёт записи локально в `mlruns/` (в Git игнорируется). Для проверки
прогонов: `mlflow ui --backend-store-uri ./mlruns`.

## Файлы, необходимые для сдачи

- `reports/LAB1/lab1_report.md` — отчёт, автоматически создаётся из **реальных** данных;
- `reports/LAB1/eda_stats.csv` — метрики, `eda_terms.csv` — слова по темам;
- `reports/LAB1/figures/*.png` — 5 графиков;
- `reports/LAB1/precommit_config.yaml` — копия `.pre-commit-config.yaml`;
- `reports/LAB1/precommit_block.log` — 3 реальных отклонённых `git commit`;
- `reports/LAB1/hash_manifest.json` — SHA256 и размеры сырого TRAIN/TEST;
- `reports/LAB1/hash_negative_test.log` — подтверждение обнаружения подмены;
- `reports/LAB1/eda_repeat.log` — проверка воспроизводимости.

**Важно:** raw CSV не кладутся в Git/архив исходного кода — это защищает от
случайного коммита большого датасета. После скачивания всё восстанавливается
по `python -m lab1.download`. Если нужен архив вместе с датасетом — скопируйте
в папку проекта `data/raw/` после скачивания.

## Git/CI/защита ветки

```bash
git init -b main
git config user.name "Ваше ФИО"
git config user.email "your@email.example"
python scripts/install_local_hook.py  # либо `pre-commit install`
git add .
git commit -m "feat(lab1): add AG News EDA and quality gates"
git branch develop
```

Запушьте репозиторий на GitHub. `Settings → Branches → Add branch protection rule`
для `main`: required pull request, at least one approval, required CI status checks.
Эти серверные настройки невозможно включить в локальном Git-архиве.

## Состояние поставляемой сборки

Если полный AG News нельзя скачать в среде сборки, **не выдумываются** хеши,
размер словаря, медианная длина, токен-частоты и MLflow run ID.
Отчёт и графики для полных данных появляются после команды `run_all.py`.
`tests/test_lab1.py` использует маленький **тестовый** корпус только для
проверки работоспособности кода; его метрики запрещено выдавать за AG News.

## Источники и политика использования

- https://huggingface.co/datasets/fancyzhx/ag_news
- Zhang, Zhao, LeCun (2015), *Character-level Convolutional Networks for Text Classification*.
- По карточке `fancyzhx/ag_news` license = `unknown`, поэтому тексты не
  перераспространяются до уточнения лицензии оригинального корпуса.

В подготовке кода и отчёта использовался ChatGPT (GPT-6). Сведения для
`defense_checklist.md` включены в проект.
