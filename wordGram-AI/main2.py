from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re
import httpx
import difflib
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch

# Инициализация
print("=" * 60)
print("WordGram AI Backend - Инициализация")
print("=" * 60)

# Определяем путь к обученной модели
# Модель находится в ../wordGram-AiTraining/models/grammar_corrector
current_dir = Path(__file__).parent
model_path = current_dir.parent / "wordGram-AiTraining" / "models" / "grammar_corrector"
model_path = model_path.resolve()

# Префикс, использованный при обучении модели
TRAINING_PREFIX = "Исправить грамматику и пунктуацию: "

# Загрузка модели для исправления текста
print("\n[1/2] Загрузка обученной модели исправления текста...")
print(f"   Путь к модели: {model_path}")

# Проверяем существование модели
if not model_path.exists():
    print(f"\n❌ Ошибка: Модель не найдена по пути: {model_path}")
    print("   Убедитесь, что модель была обучена и находится в указанной директории.")
    print("   Альтернатива: используйте базовую модель ai-forever/ruT5-large")
    print("\n   Попытка загрузить базовую модель...")
    try:
        text_correction_tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-large")
        text_correction_model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/ruT5-large")
        
        # Перемещаем модель на GPU если доступен
        if torch.cuda.is_available():
            print("   Перемещение модели на GPU...")
            text_correction_model = text_correction_model.to("cuda")
            print(f"   Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
        
        text_correction_model.eval()
        print("✓ Базовая модель загружена успешно (обученная модель не найдена)")
        USE_TRAINED_MODEL = False
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке базовой модели: {e}")
        exit(1)
else:
    try:
        print("   Загрузка токенизатора...")
        text_correction_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        print("   Загрузка обученной модели...")
        text_correction_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
        
        # Перемещаем модель на GPU если доступен
        if torch.cuda.is_available():
            print("   Перемещение модели на GPU...")
            text_correction_model = text_correction_model.to("cuda")
            print(f"   Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   GPU недоступен, используется CPU")
        
        text_correction_model.eval()
        print("✓ Обученная модель загружена успешно!")
        USE_TRAINED_MODEL = True
    except KeyboardInterrupt:
        print("\n\n❌ Загрузка модели прервана пользователем.")
        print("   Пожалуйста, запустите сервер снова.")
        exit(1)
    except Exception as e:
        print(f"\n⚠ Ошибка при загрузке обученной модели: {e}")
        print("   Попытка загрузить базовую модель...")
        try:
            text_correction_tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-large")
            text_correction_model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/ruT5-large")
            
            # Перемещаем модель на GPU если доступен
            if torch.cuda.is_available():
                print("   Перемещение модели на GPU...")
                text_correction_model = text_correction_model.to("cuda")
                print(f"   Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
            
            text_correction_model.eval()
            print("✓ Базовая модель загружена успешно (обученная модель не загрузилась)")
            USE_TRAINED_MODEL = False
        except Exception as e2:
            print(f"\n❌ Ошибка при загрузке базовой модели: {e2}")
            exit(1)

print("\n[2/2] Система исправления текста готова к работе")
print("=" * 60)
print("✓ Все компоненты загружены. Запуск сервера...")
print("=" * 60 + "\n")

app = FastAPI(title="WordGram Spell Check API (Yandex Speller)")

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

# Yandex Speller API URL
YANDEX_SPELLER_URL = "https://speller.yandex.net/services/spellservice.json/checkText"

def correct_text_with_ai(text: str) -> str:
    """
    Исправляет орфографические и пунктуационные ошибки в тексте используя обученную модель
    
    Args:
        text: Текст с возможными ошибками
    
    Returns:
        Исправленный текст
    """
    if not text or not text.strip():
        return text
    
    try:
        # Используем префикс, который использовался при обучении модели
        if USE_TRAINED_MODEL:
            # Для обученной модели используем тот же префикс, что и при обучении
            prompt = TRAINING_PREFIX + text
        else:
            # Для базовой модели используем более детальный промпт
            prompt = f"Ты профессиональный корректор с обширными познаниями в русской филологии. Вычитай предоставленный текст, исправь орфографические, грамматические и пунктуационные ошибки. Ты не должен ничего дописывать или перефразировать. Тебе неодходимо выводить только исправленный текст который помечен тегами \"<TEXT>\" вот так: <TEXT> Сам текст <TEXT>.\n\nТекст: <TEXT> {text} <TEXT>"
        
        # Логирование перед отправкой запроса
        print(f"\n[AI Correction] Отправка запроса к модели:")
        print(f"  Исходный текст: {text}")
        print(f"  Длина текста: {len(text)} символов")
        print(f"  Длина промпта: {len(prompt)} символов")
        
        # Токенизация
        print(f"  Токенизация промпта...")
        # Определяем максимальную длину в зависимости от типа модели
        max_input_length = 256 if USE_TRAINED_MODEL else 512
        inputs = text_correction_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        )
        print(f"  Размер входных токенов: {inputs['input_ids'].shape}")
        
        # Перемещаем входные данные на то же устройство, что и модель
        device = next(text_correction_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Генерация с параметрами для T5
        print(f"  Генерация исправленного текста (устройство: {device})...")
        # Параметры генерации зависят от типа модели
        if USE_TRAINED_MODEL:
            # Для обученной модели используем параметры, близкие к тем, что использовались при обучении
            max_output_length = min(256, len(text) + 50)  # Ограничиваем длину
            generation_params = {
                "max_length": max_output_length,
                "num_beams": 3,  # Как в config: num_beams=3
                "early_stopping": True,
                "no_repeat_ngram_size": 2,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "do_sample": False,
            }
        else:
            # Для базовой модели используем более агрессивные параметры
            generation_params = {
                "max_length": len(text) + 150,
                "min_length": max(1, len(text) // 3),
                "num_beams": 5,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "do_sample": False,
            }
        
        with torch.no_grad():
            outputs = text_correction_model.generate(
                inputs["input_ids"],
                **generation_params
            )
        print(f"  Генерация завершена. Размер выходных токенов: {outputs.shape}")
        
        # Декодирование результата
        print(f"  Декодирование результата...")
        corrected_text = text_correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Сырой результат модели: {corrected_text[:200]}..." if len(corrected_text) > 200 else f"  Сырой результат модели: {corrected_text}")
        
        # Обработка результата в зависимости от типа модели
        if USE_TRAINED_MODEL:
            # Обученная модель уже обучена на правильном формате, просто убираем префикс если он есть
            if corrected_text.startswith(TRAINING_PREFIX):
                corrected_text = corrected_text[len(TRAINING_PREFIX):].strip()
        else:
            # Для базовой модели нужно больше обработки
            # Убираем возможные префиксы промпта из результата
            # Ищем начало текста после промпта
            text_marker = "Текст:"
            if text_marker in corrected_text:
                corrected_text = corrected_text.split(text_marker, 1)[-1].strip()
            
            # Также убираем возможные повторения инструкции
            instruction_phrases = [
                "Ты профессиональный корректор",
                "Вычитай предоставленный текст",
                "исправленный текст",
                "Исправленный текст:",
                "Исправленный текст",
            ]
            
            for phrase in instruction_phrases:
                if corrected_text.startswith(phrase):
                    # Находим где заканчивается инструкция и начинается текст
                    # Обычно после двоеточия или новой строки
                    if ":" in corrected_text:
                        corrected_text = corrected_text.split(":", 1)[-1].strip()
                    elif "\n" in corrected_text:
                        corrected_text = corrected_text.split("\n", 1)[-1].strip()
                    break
            
            # Убираем лишние фразы, которые модель может добавить
            # Удаляем фразы типа "Ты не должен ничего дописывать или переписывать"
            unwanted_phrases = [
                "Ты не должен ничего дописывать или переписывать",
                "Ты не должен ничего дописывать или перефразировать",
                "не должен ничего дописывать",
                "не должен переписывать",
                "не должен перефразировать",
            ]
            
            for phrase in unwanted_phrases:
                # Удаляем фразу и все что после неё, если она найдена
                if phrase.lower() in corrected_text.lower():
                    # Находим позицию фразы
                    idx = corrected_text.lower().find(phrase.lower())
                    if idx > 0:
                        # Берем только текст до этой фразы
                        corrected_text = corrected_text[:idx].strip()
                        # Убираем возможные знаки препинания в конце
                        corrected_text = re.sub(r'[.,!?:;]+$', '', corrected_text).strip()
                        break
        
        # Очистка и нормализация
        corrected_text = corrected_text.strip()
        # Нормализуем пробелы (но сохраняем одиночные пробелы)
        corrected_text = re.sub(r'\s+', ' ', corrected_text)
        
        # Убираем множественные знаки препинания (например, ",,,," -> ",")
        corrected_text = re.sub(r'([.,!?:;])\1+', r'\1', corrected_text)
        
        # Убираем знаки препинания в начале (кроме точек в конце предложения)
        corrected_text = re.sub(r'^[.,!?:;]+', '', corrected_text)
        
        # Всегда возвращаем результат без проверок
        print(f"  Очищенный результат: {corrected_text[:200]}..." if len(corrected_text) > 200 else f"  Очищенный результат: {corrected_text}")
        print(f"  Длина результата: {len(corrected_text)} символов (исходный: {len(text)})")
        print(f"[AI Correction] Запрос обработан успешно\n")
        
        return corrected_text
    
    except Exception as e:
        print(f"Warning: Error correcting text with AI: {e}")
        import traceback
        print(traceback.format_exc())
        # В случае ошибки возвращаем исходный текст
        return text

def find_errors_with_ai(original: str) -> List[dict]:
    """
    Находит орфографические и пунктуационные ошибки используя ИИ модель для анализа контекста
    
    Args:
        original: Оригинальный текст
    
    Returns:
        Список ошибок (орфография + пунктуация)
    """
    errors = []
    
    if not original or not original.strip():
        return errors
    
    try:
        # Используем модель для анализа всего текста целиком
        # Это позволяет модели лучше понимать контекст
        corrected_text = correct_text_with_ai(original)
        
        # Если текст не изменился, ошибок нет
        if corrected_text == original:
            return errors
        
        # Используем difflib для нахождения различий на уровне символов
        # Это более надежный способ найти все изменения
        matcher = difflib.SequenceMatcher(None, original, corrected_text)
        found_positions = set()
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                # Текст был заменен
                orig_segment = original[i1:i2]
                corr_segment = corrected_text[j1:j2]
                
                # Проверяем, является ли сегмент уже полным словом/фразой
                # (окружен пробелами или знаками препинания)
                is_complete_word = (
                    (i1 == 0 or original[i1 - 1] in ' \n\t.,!?:;') and
                    (i2 >= len(original) or original[i2] in ' \n\t.,!?:;')
                )
                
                if is_complete_word:
                    # Сегмент уже является полным словом/фразой, используем его как есть
                    start = i1
                    end = i2
                    orig_word = orig_segment
                    corr_word = corr_segment
                else:
                    # Расширяем сегмент до границ слов
                    start = i1
                    end = i2
                    
                    # Расширяем назад до начала слова
                    while start > 0 and original[start - 1] not in ' \n\t':
                        start -= 1
                    
                    # Расширяем вперед до конца слова
                    while end < len(original) and original[end] not in ' \n\t':
                        end += 1
                    
                    # Извлекаем полное слово/фразу
                    orig_word = original[start:end]
                    
                    # Находим соответствующую часть в исправленном тексте
                    # Используем позицию относительно начала замены
                    corr_start = j1
                    corr_end = j2
                    
                    # Расширяем до границ слова в исправленном тексте
                    while corr_start > 0 and corrected_text[corr_start - 1] not in ' \n\t':
                        corr_start -= 1
                    while corr_end < len(corrected_text) and corrected_text[corr_end] not in ' \n\t':
                        corr_end += 1
                    
                    corr_word = corrected_text[corr_start:corr_end]
                
                pos_key = (start, end)
                if pos_key not in found_positions:
                    errors.append({
                        "word": orig_word,
                        "position": {"start": start, "end": end},
                        "suggestions": [corr_word],
                        "severity": "error"
                    })
                    found_positions.add(pos_key)
            
            elif tag == 'insert':
                # В исправленном тексте добавлены символы (например, знаки препинания)
                # Находим позицию вставки в оригинале
                insert_pos = i1
                
                # Определяем, что было добавлено
                added_text = corrected_text[j1:j2]
                
                # Если это только знаки препинания, вставляем их после предыдущего слова
                if re.match(r'^[.,!?:;]+$', added_text):
                    pos_key = (insert_pos, insert_pos)
                    if pos_key not in found_positions:
                        # Находим слово перед позицией вставки
                        word_before = ""
                        word_start = insert_pos
                        while word_start > 0 and original[word_start - 1] not in ' \n\t':
                            word_start -= 1
                        if word_start < insert_pos:
                            word_before = original[word_start:insert_pos]
                        
                        errors.append({
                            "word": word_before if word_before else original[max(0, insert_pos-5):insert_pos],
                            "position": {"start": insert_pos, "end": insert_pos},
                            "suggestions": [added_text],
                            "severity": "error"
                        })
                        found_positions.add(pos_key)
                else:
                    # Добавлен текст (например, пробел или слово)
                    pos_key = (insert_pos, insert_pos)
                    if pos_key not in found_positions:
                        errors.append({
                            "word": original[max(0, insert_pos-1):insert_pos] if insert_pos > 0 else "",
                            "position": {"start": insert_pos, "end": insert_pos},
                            "suggestions": [added_text],
                            "severity": "error"
                        })
                        found_positions.add(pos_key)
            
            elif tag == 'delete':
                # В оригинале были символы, которых нет в исправленном тексте
                deleted_text = original[i1:i2]
                pos_key = (i1, i2)
                if pos_key not in found_positions:
                    errors.append({
                        "word": deleted_text,
                        "position": {"start": i1, "end": i2},
                        "suggestions": [""],  # Удаление
                        "severity": "error"
                    })
                    found_positions.add(pos_key)
        
        return errors
    
    except Exception as e:
        print(f"Warning: Error in find_errors_with_ai: {e}")
        import traceback
        print(traceback.format_exc())
        return errors

def find_errors(original: str, corrected: str) -> List[dict]:
    """
    Находит орфографические и пунктуационные ошибки используя ИИ для анализа контекста
    
    Args:
        original: Оригинальный текст
        corrected: Текст с исправлениями (не используется, оставлен для совместимости)
    
    Returns:
        Список ошибок (без дубликатов)
    """
    # Используем ИИ для анализа текста напрямую
    return find_errors_with_ai(original)

async def check_text_with_yandex(text: str, language: str = "ru") -> List[dict]:
    """
    Проверяет текст через Yandex Speller API
    
    Args:
        text: Текст для проверки
        language: Язык проверки (ru, en, uk)
    
    Returns:
        Список ошибок в формате Yandex API
    """
    try:
        # Параметры запроса
        params = {
            "text": text,
            "lang": language,
            "options": 0  # 0 - обычная проверка, можно добавить опции
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(YANDEX_SPELLER_URL, params=params)
            response.raise_for_status()
            return response.json()
    
    except httpx.TimeoutException:
        print("Warning: Yandex Speller API timeout")
        return []
    except httpx.HTTPStatusError as e:
        print(f"Warning: Yandex Speller API HTTP error: {e}")
        return []
    except Exception as e:
        print(f"Warning: Error calling Yandex Speller API: {e}")
        return []

def map_yandex_errors_to_positions(yandex_errors: List[dict], original_text: str) -> List[dict]:
    """
    Преобразует ошибки Yandex API в формат с позициями в тексте
    
    Args:
        yandex_errors: Список ошибок от Yandex API
        original_text: Исходный текст
    
    Returns:
        Список ошибок с позициями в формате приложения
    """
    errors = []
    
    for yandex_error in yandex_errors:
        word = yandex_error.get("word", "")
        suggestions = yandex_error.get("s", [])
        pos = yandex_error.get("pos", 0)
        length = yandex_error.get("len", len(word))
        
        # Yandex API возвращает позицию в символах (с учетом пробелов)
        # Находим точные границы слова в тексте
        start = pos
        end = pos + length
        
        # Убеждаемся, что позиции не выходят за границы текста
        start = max(0, min(start, len(original_text)))
        end = max(start, min(end, len(original_text)))
        
        # Извлекаем слово из текста для точного совпадения
        actual_word = original_text[start:end]
        
        if suggestions:
            errors.append({
                "word": actual_word,
                "position": {"start": start, "end": end},
                "suggestions": suggestions[:5],  # Ограничиваем до 5 предложений
                "severity": "error"
            })
    
    return errors

@app.get("/")
def root():
    model_info = {
        "type": "trained" if USE_TRAINED_MODEL else "base",
        "path": str(model_path) if USE_TRAINED_MODEL else "ai-forever/ruT5-large",
        "prefix": TRAINING_PREFIX if USE_TRAINED_MODEL else "custom_prompt"
    }
    return {
        "message": "WordGram Spell Check API (AI Text Correction)", 
        "status": "running",
        "model": model_info,
        "features": ["spelling_check", "punctuation_restoration", "context_aware_correction"],
        "device": str(next(text_correction_model.parameters()).device) if hasattr(text_correction_model, 'parameters') else "unknown"
    }

@app.post("/api/spell-check", response_model=SpellCheckResponse)
async def spell_check(request: SpellCheckRequest):
    try:
        if not request.text or not request.text.strip():
            return SpellCheckResponse(errors=[], correctedText=request.text)
        
        original_text = request.text
        
        # Используем ИИ модель для исправления текста (орфография + пунктуация)
        # Модель учитывает контекст и правильно исправляет текст
        corrected_text = correct_text_with_ai(original_text)
        
        # Находим все ошибки (орфография + пунктуация) через сравнение с исправленным текстом
        all_errors = find_errors(original_text, corrected_text)
        
        # Используем результат модели напрямую, так как он уже учитывает контекст
        # и правильно обрабатывает пробелы и пунктуацию
        final_corrected_text = corrected_text
        
        # Очищаем финальный текст от множественных знаков препинания (на всякий случай)
        final_corrected_text = re.sub(r'([.,!?:;])\1+', r'\1', final_corrected_text)
        final_corrected_text = re.sub(r'\s+', ' ', final_corrected_text).strip()
        
        # Конвертируем в SpellError объекты
        spell_errors = [
            SpellError(
                word=err["word"],
                position=err["position"],
                suggestions=err["suggestions"],
                severity=err.get("severity", "error")
            )
            for err in all_errors
        ]
        
        return SpellCheckResponse(
            errors=spell_errors,
            correctedText=final_corrected_text if final_corrected_text != original_text else None
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
    print("   Используется:")
    if USE_TRAINED_MODEL:
        print(f"   - Обученная модель: {model_path}")
        print("   - Префикс: 'Исправить грамматику и пунктуацию: '")
    else:
        print("   - Базовая модель: ai-forever/ruT5-large (fallback)")
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

