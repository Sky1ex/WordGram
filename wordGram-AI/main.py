from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
import torch

# Load model and tokenizer
print("=" * 60)
print("WordGram AI Backend - Загрузка модели")
print("=" * 60)
print("\n[1/2] Загрузка токенизатора...")
try:
    # Попробуем загрузить fast токенизатор (по умолчанию)
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-large")
    print("✓ Токенизатор загружен успешно (fast)")
except Exception as e:
    print(f"⚠ Не удалось загрузить fast токенизатор: {e}")
    print("Попытка загрузить медленный токенизатор...")
    # Используем медленный токенизатор, если fast не работает
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-large", use_fast=False)
    print("✓ Токенизатор загружен успешно (slow)")

print("\n[2/2] Загрузка модели ruT5-large...")
print("⚠ ВНИМАНИЕ: При первом запуске модель будет скачана из Hugging Face (~3GB)")
print("   Это может занять несколько минут в зависимости от скорости интернета.")
print("   Пожалуйста, не прерывайте процесс загрузки!\n")

try:
    model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/ruT5-large")
    print("\n✓ Модель загружена успешно!")
except KeyboardInterrupt:
    print("\n\n❌ Загрузка модели прервана пользователем.")
    print("   Пожалуйста, запустите сервер снова - загрузка продолжится с того места, где остановилась.")
    exit(1)
except Exception as e:
    print(f"\n❌ Ошибка при загрузке модели: {e}")
    print("   Проверьте интернет-соединение и попробуйте снова.")
    exit(1)

# Set model to evaluation mode
model.eval()

print("\n" + "=" * 60)
print("✓ Все компоненты загружены. Запуск сервера...")
print("=" * 60 + "\n")

app = FastAPI(title="WordGram Spell Check API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SpellCheckRequest(BaseModel):
    text: str
    language: Optional[str] = "ru"

class SpellError(BaseModel):
    word: str
    position: dict
    suggestions: List[str]
    severity: Optional[str] = "error"

class SpellCheckResponse(BaseModel):
    errors: List[SpellError]
    correctedText: Optional[str] = None

def correct_word_with_model(word: str) -> Optional[str]:
    """Исправляет одно слово используя ruT5 модель с правильными префиксами задач T5"""
    # Очищаем слово от знаков препинания для проверки
    word_clean = word.strip('.,!?;:"()[]{}')
    
    # Если слово слишком короткое или состоит только из знаков препинания, пропускаем
    if len(word_clean) < 2:
        return None
    
    # T5 модели используют префиксы задач. Пробуем разные варианты префиксов
    # для задач, которые могут помочь с исправлением орфографии
    prompts = [
        # Прямое исправление орфографии
        f"исправить орфографию: {word_clean}",
        # f"spell correct: {word_clean}",
        # Парафразирование (может исправить ошибки)
        # f"перефразировать: {word_clean}",
        # f"paraphrase: {word_clean}",
        # # Исправление текста
        # f"исправить текст: {word_clean}",
        # f"correct text: {word_clean}",
        # # Простой вариант
        # f"исправить: {word_clean}",
    ]
    
    for prompt in prompts:
        try:
            # Токенизация с правильными параметрами для T5
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            )
            
            # Генерация с параметрами, рекомендованными для T5
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=32,  # Для одного слова достаточно короткого вывода
                    min_length=1,
                    num_beams=3,  # Небольшое количество лучей для скорости
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.2,  # Штраф за повторения
                    length_penalty=0.6,  # Предпочтение более коротким ответам
                    do_sample=False,  # Детерминированный режим
                )
            
            # Декодирование
            corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected = corrected.strip()
            
            # Убираем возможные префиксы
            prefixes_to_remove = [
                "исправить орфографию:",
                "spell correct:",
                "перефразировать:",
                "paraphrase:",
                "исправить текст:",
                "correct text:",
                "исправить:",
                "Исправить орфографию:",
                "Исправить слово:",
                "Орфография:",
                "Исправь:",
                "Исправить:",
                "correction:",
            ]
            
            for prefix in prefixes_to_remove:
                if corrected.lower().startswith(prefix.lower()):
                    corrected = corrected[len(prefix):].strip()
                    break
            
            # Фильтруем некорректные результаты
            # Если результат содержит много точек/запятых - пропускаем
            if re.search(r'[.,]{2,}', corrected):
                continue
            
            # Если результат слишком длинный (больше чем в 2 раза длиннее оригинала) - пропускаем
            if len(corrected) > len(word_clean) * 2 + 5:  # Небольшой запас
                continue
            
            # Если результат содержит пробелы (не одно слово) - пропускаем
            if ' ' in corrected:
                continue
            
            # Если результат пустой - пропускаем
            if not corrected:
                continue
            
            # Если результат равен оригиналу - слово правильное
            if corrected.lower() == word_clean.lower():
                return None
            
            # Проверяем, что результат - разумное слово (только буквы)
            if re.match(r'^[а-яА-ЯёЁa-zA-Z]+$', corrected):
                # Дополнительная проверка: результат должен быть похож на оригинал
                # (не слишком отличаться по длине)
                if abs(len(corrected) - len(word_clean)) <= len(word_clean) * 0.5 + 2:
                    return corrected
                
        except Exception as e:
            # Логируем ошибку для отладки
            print(f"Warning: Error with prompt '{prompt}': {e}")
            continue
    
    # Если ни один промпт не дал хорошего результата, возвращаем None
    return None

@app.get("/")
def root():
    return {"message": "WordGram Spell Check API", "status": "running"}

@app.post("/api/spell-check", response_model=SpellCheckResponse)
async def spell_check(request: SpellCheckRequest):
    try:
        if not request.text or not request.text.strip():
            return SpellCheckResponse(errors=[], correctedText=request.text)
        
        original_text = request.text
        errors = []
        
        # Проверяем каждое слово отдельно через модель
        words_with_positions = []
        for match in re.finditer(r'\S+', original_text):
            word = match.group()
            words_with_positions.append({
                'word': word,
                'start': match.start(),
                'end': match.end()
            })
        
        # Проверяем каждое слово через модель
        corrected_text = original_text
        for item in words_with_positions:
            word = item['word']
            corrected = correct_word_with_model(word)
            
            if corrected and corrected.lower() != word.lower():
                # Найдена ошибка
                errors.append({
                    "word": word,
                    "position": {"start": item['start'], "end": item['end']},
                    "suggestions": [corrected],
                    "severity": "error"
                })
        
        # Формируем исправленный текст, заменяя слова с ошибками
        if errors:
            corrected_text = original_text
            # Заменяем слова в обратном порядке, чтобы позиции не сдвигались
            for error in sorted(errors, key=lambda x: x['position']['start'], reverse=True):
                if error['suggestions']:
                    suggestion = error['suggestions'][0]
                    start = error['position']['start']
                    end = error['position']['end']
                    corrected_text = corrected_text[:start] + suggestion + corrected_text[end:]
        
        # Конвертируем в SpellError объекты
        spell_errors = [
            SpellError(
                word=err["word"],
                position=err["position"],
                suggestions=err["suggestions"],
                severity=err.get("severity", "error")
            )
            for err in errors
        ]
        
        return SpellCheckResponse(
            errors=spell_errors,
            correctedText=corrected_text if corrected_text != original_text else None
        )
    except Exception as e:
        import traceback
        print(f"Error in spell_check: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import socket
    
    def is_port_available(port):
        """Проверяет, доступен ли порт"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except OSError:
                return False
    
    # Пробуем разные порты, начиная с 8000
    port = 8000
    max_attempts = 10
    
    for attempt in range(max_attempts):
        if is_port_available(port):
            break
        else:
            if attempt == 0:
                print(f"\n⚠ Порт {port} занят, ищу свободный порт...")
            port += 1
    
    if not is_port_available(port):
        print(f"\n❌ Не удалось найти свободный порт в диапазоне 8000-{port}")
        print("   Закройте процессы, использующие эти порты, или измените порт вручную")
        exit(1)
    
    if port != 8000:
        print(f"✓ Используется порт {port} (8000 занят)")
    
    print(f"\n🌐 Сервер запущен на http://localhost:{port}")
    print(f"   API доступен по адресу: http://localhost:{port}/api/spell-check")
    print("   Нажмите Ctrl+C для остановки\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            print(f"\n❌ Ошибка: Порт {port} занят!")
            print("   Возможные решения:")
            print("   1. Закройте другой процесс, использующий этот порт")
            print("   2. Найдите процесс: netstat -ano | findstr :8000")
            print("   3. Завершите процесс: taskkill /PID <номер_процесса> /F")
            print(f"   4. Или измените порт в коде (строка uvicorn.run)")
        else:
            print(f"\n❌ Ошибка запуска сервера: {e}")
        exit(1)