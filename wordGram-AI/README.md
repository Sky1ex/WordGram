# WordGram AI Backend

Бэкенд сервер для проверки орфографии с использованием модели ruT5-large.

## Установка

### Требования
- Python 3.8 или выше
- pip

### Шаги установки

1. Перейдите в папку проекта:
```bash
cd wordGram-AI
```

2. Создайте виртуальное окружение (рекомендуется):
```bash
python -m venv venv
```

3. Активируйте виртуальное окружение:
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

4. Установите зависимости:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Примечание:** Если возникают проблемы с установкой `pydantic-core`:
- Убедитесь, что используете Python 3.8-3.12 (Python 3.13 может требовать Rust)
- Или используйте предкомпилированные пакеты:
  ```bash
  pip install --only-binary :all: -r requirements.txt
  ```

**Важно:** Убедитесь, что установлены все зависимости для токенизатора:
```bash
pip install tiktoken sentencepiece
```

## Запуск

Запустите сервер:
```bash
python main.py
```

Сервер будет доступен по адресу: `http://localhost:8000`

При первом запуске модель будет загружена из Hugging Face (это может занять несколько минут).

## API

### POST /api/spell-check

Проверяет орфографию в тексте.

**Request:**
```json
{
  "text": "Привет, как дела?",
  "language": "ru"
}
```

**Response:**
```json
{
  "errors": [
    {
      "word": "Привет",
      "position": {"start": 0, "end": 6},
      "suggestions": ["Привет"],
      "severity": "error"
    }
  ],
  "correctedText": "Привет, как дела?"
}
```

### GET /

Проверка работы сервера:
```bash
curl http://localhost:8000/
```
