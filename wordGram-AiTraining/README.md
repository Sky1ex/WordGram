# WordGram AI Training

Этот проект содержит код для обучения модели исправления грамматических и пунктуационных ошибок для русского языка.

## Структура проекта

```
wordGram-AiTraining/
├── config.py           # Конфигурация обучения
├── data_utils.py       # Утилиты для работы с данными
├── download_dataset.py # Скрипт загрузки датасета с GitHub
├── prepare_data.py     # Скрипт подготовки данных
├── train.py            # Основной скрипт обучения
├── evaluate.py         # Скрипт оценки модели
├── test_model.py       # Скрипт быстрого тестирования модели
├── requirements.txt    # Зависимости
└── README.md          # Документация
```

## Установка

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас установлен PyTorch с поддержкой CUDA (если используете GPU):

```bash
# Для CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для CPU
pip install torch torchvision torchaudio
```

## Подготовка данных

### Использование датасетов из папки spellCheck

Если вы добавили датасеты в папку `data/spellCheck`, вы можете использовать их для обучения:

```bash
# Подготовка данных только из spellCheck датасетов
python prepare_data.py --spellcheck-only

# Или указать путь к папке spellCheck
python prepare_data.py --spellcheck-only --spellcheck data/spellCheck
```

Скрипт автоматически найдет все JSON файлы в папке `spellCheck` и её подпапках (например, `GithubTypoCorpusRu/test.json`, `MedSpellChecker/test.json`, `RUSpellRU/test.json`).

**Формат данных в JSON файлах:**
Каждая строка должна содержать JSON объект с полями:
- `source`: неправильный текст
- `correction`: правильный текст
- `domain`: источник данных (опционально)

Пример:
```json
{"source": "привет как дела", "correction": "Привет, как дела?", "domain": "GithubTypoCorpusRu"}
```

### Объединение данных из разных источников

Вы можете объединить данные из spellCheck с другими датасетами:

```bash
# Объединение данных из spellCheck и CSV файла
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv --combine

# Если папка spellCheck находится в другом месте
python prepare_data.py --input data/your_dataset.csv --format csv --spellcheck path/to/spellCheck --combine
```

### Использование Russian Grammar Error Correction Dataset

Рекомендуется использовать готовый датасет [Russian Grammar Error Correction Dataset](https://github.com/dreuxx/Russian-Grammar-Error-Correction-Dataset) (25,362 пар предложений).

#### Автоматическая загрузка датасета

```bash
# Загрузка датасета с GitHub
python download_dataset.py

# Если файл уже существует и нужно перезаписать
python download_dataset.py --force

# Датасет будет загружен в папку data/
```

#### Подготовка данных из загруженного датасета

```bash
# Подготовка данных только из CSV файла
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv

# Подготовка данных из CSV файла + объединение с spellCheck (если папка существует)
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv

# Или если файл уже загружен с другим именем
python prepare_data.py --input data/your_dataset.csv --format csv
```

Это создаст файлы `data/train.jsonl`, `data/val.jsonl` и `data/test.jsonl` с разделением 80/10/10.

#### Ручная загрузка датасета

1. Перейдите на [GitHub репозиторий](https://github.com/dreuxx/Russian-Grammar-Error-Correction-Dataset)
2. Скачайте файл `russian_gec_dataset_final (1).csv`
3. Поместите его в папку `data/`
4. Запустите подготовку данных:

```bash
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv
```

### Формат данных

Проект поддерживает несколько форматов данных:

#### JSONL формат
Каждая строка содержит JSON объект с полями:
- `input`: текст с ошибками
- `target`: исправленный текст

Пример:
```json
{"input": "привет как дела", "target": "Привет, как дела?"}
{"input": "я иду в магазин купить хлеб", "target": "Я иду в магазин купить хлеб."}
```

#### CSV формат (Russian GEC Dataset)
Файл должен содержать столбцы:
- `incorrect` / `correct` - неправильный и правильный текст
- или `input_text` / `target_text` - входной и целевой текст

#### Создание примера датасета

Для быстрого тестирования можно создать небольшой пример датасета:

```bash
python prepare_data.py
```

Это создаст файлы `data/train.jsonl`, `data/val.jsonl` и `data/test.jsonl` с примерами данных.

### Подготовка своих данных

Если у вас есть файл с данными, используйте:

```bash
# Для JSONL файла
python prepare_data.py --input your_data.jsonl --output data

# Для JSON файла (должен содержать списки "input" и "target")
python prepare_data.py --input your_data.json --output data --format json

# Для CSV файла (должен содержать колонки "input"/"target" или "incorrect"/"correct")
python prepare_data.py --input your_data.csv --output data --format csv
```

## Обучение модели

### Базовая конфигурация

Перед обучением вы можете изменить параметры в `config.py`:

```python
# Модель для обучения
model_name: str = "ai-forever/ruT5-base"  # или "ai-forever/ruT5-large"

# Параметры обучения
batch_size: int = 16
learning_rate: float = 5e-5
num_train_epochs: int = 3
```

### Запуск обучения

```bash
python train.py
```

Модель будет сохранена в директории, указанной в `config.output_dir` (по умолчанию: `models/grammar_corrector`).

### Рекомендации по обучению

1. **Размер датасета**: Рекомендуется использовать не менее 1000-10000 примеров для качественного обучения.

2. **Выбор модели**:
   - `ai-forever/ruT5-base` - быстрее обучается, требует меньше памяти
   - `ai-forever/ruT5-large` - лучшее качество, но требует больше памяти и времени

3. **GPU память**: Если возникают проблемы с памятью GPU:
   - Уменьшите `batch_size` в `config.py`
   - Увеличьте `gradient_accumulation_steps`
   - Используйте `fp16=True` для mixed precision

4. **Продолжение обучения**: Чтобы продолжить обучение с чекпоинта:
   ```python
   # В config.py измените:
   output_dir: str = "models/grammar_corrector"  # путь к существующей модели
   ```

## Оценка модели

После обучения оцените модель:

```bash
python evaluate.py --model models/grammar_corrector
```

Для оценки на конкретном тестовом наборе:

```bash
python evaluate.py --model models/grammar_corrector --test-data data/test.jsonl
```

## Быстрое тестирование модели

Для быстрого тестирования модели на отдельных примерах используйте:

```bash
# Тестирование с примерами по умолчанию
python test_model.py --model models/grammar_corrector

# Тестирование на конкретном тексте
python test_model.py --model models/grammar_corrector --text "привет как дела"

# Тестирование на нескольких текстах
python test_model.py --model models/grammar_corrector --text "привет как дела" --text "я иду в магазин"
```

## Использование обученной модели

После обучения модель можно использовать в основном приложении. Пример:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Загрузка модели
model_path = "models/grammar_corrector"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Создание pipeline
corrector = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Используйте -1 для CPU
)

# Исправление текста
text = "привет как дела"
prefix = "Исправить грамматику и пунктуацию: "
result = corrector(prefix + text, max_length=512, num_beams=5)

corrected = result[0]["generated_text"].replace(prefix, "").strip()
print(corrected)  # "Привет, как дела?"
```

## Использование Russian Grammar Error Correction Dataset

Этот проект оптимизирован для работы с [Russian Grammar Error Correction Dataset](https://github.com/dreuxx/Russian-Grammar-Error-Correction-Dataset), который содержит:

- **25,362 пар предложений** с грамматическими ошибками и их исправлениями
- Различные типы ошибок: согласование, спряжение, предлоги, падежи, порядок слов, пунктуация
- Средняя длина предложения: ~12 токенов
- Лицензия: CC BY 4.0 (можно использовать коммерчески с указанием автора)

### Быстрый старт с Russian GEC Dataset

```bash
# 1. Загрузка датасета
python download_dataset.py

# 2. Подготовка данных
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv

# 3. Обучение модели
python train.py

# 4. Тестирование
python test_model.py --model models/grammar_corrector
```

### Рекомендации по использованию датасета

1. **Размер датасета**: Russian GEC Dataset содержит достаточно данных для качественного обучения (25k пар)

2. **Разделение данных**: По умолчанию используется разделение 80/10/10 (train/val/test), что оптимально для этого размера датасета

3. **Дополнительные данные**: Вы можете комбинировать Russian GEC Dataset со своими данными для улучшения качества модели

4. **Типы ошибок**: Датасет покрывает основные типы грамматических ошибок русского языка, но для специфических доменов может потребоваться дополнительное обучение

## Устранение проблем

### Out of Memory (OOM)

- Уменьшите `batch_size`
- Уменьшите `max_seq_length`
- Используйте `gradient_accumulation_steps`
- Используйте меньшую модель (`ruT5-base` вместо `ruT5-large`)

### Медленное обучение

- Убедитесь, что используете GPU
- Используйте `fp16=True`
- Увеличьте `dataloader_num_workers`

### Плохое качество модели

- Увеличьте размер датасета
- Увеличьте количество эпох
- Попробуйте другую модель (`ruT5-large`)
- Проверьте качество данных

## Лицензия

Этот проект является частью WordGram проекта.
