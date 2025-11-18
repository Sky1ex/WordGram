# Использование датасетов из папки spellCheck

## Быстрый старт

Если вы добавили датасеты в папку `data/spellCheck`, просто выполните:

```bash
python prepare_data.py --spellcheck-only
```

Это автоматически:
1. Найдет все JSON файлы в папке `data/spellCheck` и её подпапках
2. Загрузит данные из всех найденных файлов
3. Разделит данные на train/val/test (80/10/10)
4. Сохранит результат в `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`

## Структура папки spellCheck

Ваша папка `spellCheck` может иметь любую структуру подпапок. Скрипт автоматически найдет все `.json` файлы:

```
data/spellCheck/
├── GithubTypoCorpusRu/
│   └── test.json
├── MedSpellChecker/
│   └── test.json
└── RUSpellRU/
    └── test.json
```

## Формат данных

Каждая строка в JSON файле должна содержать JSON объект с полями:

- `source` (обязательно) - текст с ошибками
- `correction` (обязательно) - исправленный текст
- `domain` (опционально) - источник данных

Пример:
```json
{"source": "привет как дела", "correction": "Привет, как дела?", "domain": "GithubTypoCorpusRu"}
{"source": "я иду в магазин", "correction": "Я иду в магазин.", "domain": "RUSpellRU"}
```

## Объединение с другими датасетами

Если у вас есть другие датасеты (например, CSV файл), вы можете объединить их с данными из spellCheck:

```bash
# Объединение данных из spellCheck и CSV файла
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv
```

По умолчанию, если папка `spellCheck` существует, данные будут объединены автоматически.

## Параметры командной строки

```bash
# Использовать только данные из spellCheck
python prepare_data.py --spellcheck-only

# Указать другой путь к папке spellCheck
python prepare_data.py --spellcheck-only --spellcheck path/to/spellCheck

# Объединить данные из файла и spellCheck
python prepare_data.py --input data/file.csv --format csv --combine

# Использовать только файл (игнорировать spellCheck)
python prepare_data.py --input data/file.csv --format csv
```

## Что происходит при загрузке

1. Скрипт ищет все `.json` файлы в указанной папке и её подпапках
2. Для каждого файла:
   - Читает файл построчно
   - Парсит каждую строку как JSON
   - Извлекает поля `source` и `correction`
   - Фильтрует пустые значения и пары, где source == correction
3. Объединяет все данные из всех файлов
4. Разделяет на train/val/test с соотношением 80/10/10
5. Сохраняет результат в JSONL формате

## Примеры использования

### Пример 1: Только spellCheck
```bash
python prepare_data.py --spellcheck-only
```

### Пример 2: Объединение с CSV
```bash
python prepare_data.py --input data/russian_gec_dataset_final\ \(1\).csv --format csv
```

### Пример 3: Указать путь к spellCheck
```bash
python prepare_data.py --spellcheck-only --spellcheck data/my_spellcheck_data
```

## После подготовки данных

После выполнения скрипта подготовки данных вы можете начать обучение:

```bash
python train.py
```

Скрипт обучения автоматически использует подготовленные файлы `data/train.jsonl`, `data/val.jsonl` и `data/test.jsonl`.











