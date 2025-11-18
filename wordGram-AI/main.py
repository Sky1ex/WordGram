from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
import httpx
import difflib
import os
import time
from pathlib import Path
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch

# Инициализация
print("=" * 60)
print("WordGram AI Backend - Инициализация")
print("=" * 60)

# Определяем путь к локальной модели RuM2M100-1.2B
current_dir = Path(__file__).parent
local_model_path = current_dir / "model_FRED"
local_model_path = local_model_path.resolve()

# Загрузка модели для исправления текста
print("\n[1/2] Загрузка модели RuM2M100-1.2B...")
print(f"   Путь к модели: {local_model_path}")

# Проверяем существование локальной модели
if not local_model_path.exists():
    print(f"\n❌ Ошибка: Модель не найдена по пути: {local_model_path}")
    print("   Убедитесь, что модель была скачана в папку model_FRED.")
    print("   Альтернатива: модель загрузится из HuggingFace Hub автоматически")
    print("\n   Попытка загрузить модель из HuggingFace Hub...")
    try:
        model_path = "ai-forever/RuM2M100-1.2B"
        print("   Загрузка модели (это может занять некоторое время)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Загружаем на CPU сначала
        text_correction_model = M2M100ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        # Перемещаем на нужное устройство
        if device == "cuda":
            text_correction_model = text_correction_model.to(device)
        text_correction_tokenizer = M2M100Tokenizer.from_pretrained(
            model_path,
            src_lang="ru",
            tgt_lang="ru"
        )
        USE_LOCAL_MODEL = False
        print(f"   ✓ Модель загружена из HuggingFace Hub на {device}")
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке модели из HuggingFace: {e}")
        exit(1)
else:
    print("   ✓ Локальная модель найдена, загружаем из локальной папки...")
    try:
        model_path = str(local_model_path)
        print("   Загрузка модели (это может занять некоторое время)...")
        print("   Загрузка весов модели в память...")
        
        # Определяем устройство заранее
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Загрузка модели на устройство: {device}")
        
        # Загружаем модель - пробуем разные варианты для избежания мета-тензоров
        print("   Загрузка модели (это может занять время)...")
        model_loaded = False
        
        # Вариант 1: Загрузка с pytorch_model.bin (более надежно)
        if not model_loaded:
            try:
                print("   Попытка 1: Загрузка с pytorch_model.bin...")
                text_correction_model = M2M100ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    use_safetensors=False  # Принудительно используем pytorch_model.bin
                )
                # Пробуем переместить на устройство для проверки
                test_device = "cpu"  # Всегда начинаем с CPU для проверки
                text_correction_model = text_correction_model.to(test_device)
                # Если дошли сюда, модель загружена правильно
                model_loaded = True
                print("   ✓ Модель загружена с pytorch_model.bin")
            except Exception as e1:
                print(f"   ⚠ Не удалось загрузить с pytorch_model.bin: {str(e1)[:100]}")
        
        # Вариант 2: Загрузка с safetensors
        if not model_loaded:
            try:
                print("   Попытка 2: Загрузка с safetensors...")
                text_correction_model = M2M100ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                    use_safetensors=True
                )
                # Пробуем переместить на устройство для проверки
                test_device = "cpu"
                text_correction_model = text_correction_model.to(test_device)
                model_loaded = True
                print("   ✓ Модель загружена с safetensors")
            except Exception as e2:
                print(f"   ⚠ Не удалось загрузить с safetensors: {str(e2)[:100]}")
        
        # Вариант 3: Загрузка без указания формата (автоматический выбор)
        if not model_loaded:
            try:
                print("   Попытка 3: Загрузка с автоматическим выбором формата...")
                text_correction_model = M2M100ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False
                )
                # Пробуем переместить на устройство для проверки
                test_device = "cpu"
                text_correction_model = text_correction_model.to(test_device)
                model_loaded = True
                print("   ✓ Модель загружена (автоматический формат)")
            except Exception as e3:
                print(f"   ❌ Все попытки загрузки не удались: {str(e3)[:100]}")
                raise e3
        
        # Теперь перемещаем на нужное устройство (GPU или CPU)
        if device == "cuda" and torch.cuda.is_available():
            print("   Перемещение модели на GPU...")
            try:
                text_correction_model = text_correction_model.to(device)
                print(f"   ✓ Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
            except Exception as gpu_error:
                print(f"   ⚠ Не удалось переместить на GPU: {gpu_error}")
                print("   Модель остается на CPU")
                device = "cpu"
        else:
            print("   Модель остается на CPU")
            device = "cpu"
        
        text_correction_tokenizer = M2M100Tokenizer.from_pretrained(
            model_path,
            src_lang="ru",
            tgt_lang="ru"
        )
        USE_LOCAL_MODEL = True
        print(f"   ✓ Модель загружена из локальной папки на {device}")
    except Exception as e:
        print(f"\n⚠ Ошибка при загрузке локальной модели: {e}")
        print("   Попытка загрузить модель из HuggingFace Hub...")
        try:
            model_path = "ai-forever/RuM2M100-1.2B"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Загружаем на CPU сначала
            text_correction_model = M2M100ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False
            )
            # Перемещаем на нужное устройство
            if device == "cuda":
                text_correction_model = text_correction_model.to(device)
            text_correction_tokenizer = M2M100Tokenizer.from_pretrained(
                model_path,
                src_lang="ru",
                tgt_lang="ru"
            )
            USE_LOCAL_MODEL = False
            print(f"   ✓ Модель загружена из HuggingFace Hub на {device}")
        except Exception as e2:
            print(f"\n❌ Ошибка при загрузке модели из HuggingFace: {e2}")
            exit(1)

# Модель уже загружена на нужное устройство в блоке загрузки выше
# Получаем устройство модели для дальнейшего использования
try:
    device = next(text_correction_model.parameters()).device
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"   Модель на GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   Модель на CPU")
except:
    device = "cpu"
    print("   Модель на CPU (по умолчанию)")

text_correction_model.eval()
USE_TRAINED_MODEL = False  # Используем предобученную модель, не обученную нами


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
    Исправляет орфографические и пунктуационные ошибки в тексте используя RuM2M100-1.2B модель
    
    Args:
        text: Текст с возможными ошибками
    
    Returns:
        Исправленный текст
    """
    if not text or not text.strip():
        return text
    
    try:
        # Логирование перед отправкой запроса
        print(f"\n[AI Correction] Отправка запроса к модели:")
        print(f"  Исходный текст: {text}")
        print(f"  Длина текста: {len(text)} символов")
        
        # Токенизация для M2M100 (без префикса, модель работает напрямую с текстом)
        # Согласно документации, токенизация должна быть простой
        print(f"  Токенизация текста...")
        encodings = text_correction_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        print(f"  Размер входных токенов: {encodings['input_ids'].shape}")
        print(f"  Входные токены (первые 10): {encodings['input_ids'][0, :min(10, encodings['input_ids'].shape[1])].tolist()}")
        
        # Перемещаем входные данные на то же устройство, что и модель
        model_device = next(text_correction_model.parameters()).device
        encodings = {k: v.to(model_device) for k, v in encodings.items()}
        
        # Получаем ID языка для русского (принудительный BOS токен)
        forced_bos_token_id = text_correction_tokenizer.get_lang_id("ru")
        print(f"  forced_bos_token_id (ru): {forced_bos_token_id}")
        
        # Вычисляем разумный max_length на основе длины входного текста
        # Для encoder-decoder моделей max_length означает максимальную длину выходной последовательности
        input_length = encodings['input_ids'].shape[1]
        # Исправленный текст обычно немного длиннее оригинала, но не более чем в 2 раза
        max_output_length = min(512, max(input_length + 50, 128))  # Минимум 128 токенов
        
        # Генерация согласно документации RuM2M100-1.2B
        print(f"  Генерация исправленного текста (устройство: {model_device})...")
        print(f"  Параметры: max_length={max_output_length}, input_length={input_length}")
        start_time = time.time()
        
        with torch.no_grad():
            generated_tokens = text_correction_model.generate(
                **encodings,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_output_length,
                min_length=max(1, input_length // 2),  # Минимальная длина - хотя бы половина входной длины
                num_beams=5,  # Используем beam search для лучшего качества
                do_sample=False,
                early_stopping=False,  # Отключаем early_stopping для полной генерации
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,  # Нейтральный штраф за длину
            )
        
        generation_time = time.time() - start_time
        print(f"  Генерация завершена за {generation_time:.2f} секунд. Размер выходных токенов: {generated_tokens.shape}")
        
        # Логируем первые несколько токенов для отладки
        if generated_tokens.shape[1] > 0:
            first_tokens = generated_tokens[0, :min(10, generated_tokens.shape[1])].tolist()
            print(f"  Первые токены результата: {first_tokens}")
            # Декодируем первые токены без skip_special_tokens для отладки
            first_tokens_text = text_correction_tokenizer.decode(first_tokens, skip_special_tokens=False)
            print(f"  Первые токены (текст, со спец. токенами): {first_tokens_text[:100]}")
        
        # Декодирование результата
        print(f"  Декодирование результата...")
        answer = text_correction_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Берем первый результат из списка
        if isinstance(answer, list) and len(answer) > 0:
            corrected_text = answer[0]
            # Если результат пустой, пробуем декодировать без skip_special_tokens
            if not corrected_text.strip():
                print(f"  ⚠ Результат пустой после skip_special_tokens, пробуем без него...")
                answer_no_skip = text_correction_tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                if isinstance(answer_no_skip, list) and len(answer_no_skip) > 0:
                    corrected_text = answer_no_skip[0]
                    print(f"  Результат без skip_special_tokens: {corrected_text[:200]}")
        else:
            corrected_text = text
        
        print(f"  Сырой результат модели: {corrected_text[:200]}..." if len(corrected_text) > 200 else f"  Сырой результат модели: {corrected_text}")
        
        # # Очистка и нормализация результата
        # corrected_text = corrected_text.strip()
        
        # # Нормализуем пробелы
        # corrected_text = re.sub(r'\s+', ' ', corrected_text)
        
        # # Убираем множественные знаки препинания
        # corrected_text = re.sub(r'([.,!?:;])\1+', r'\1', corrected_text)
        
        # # Убираем знаки препинания в начале
        # corrected_text = re.sub(r'^[.,!?:;]+', '', corrected_text)
        
        # Если результат пустой или слишком короткий, возвращаем исходный текст
        if not corrected_text or len(corrected_text) < len(text) * 0.3:
            print(f"  ⚠ Результат слишком короткий, возвращаем исходный текст")
            corrected_text = text
        
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

def find_text_errors_corrected(original_text: str, corrected_text: str) -> Dict[str, Any]:
    """
    Исправленный алгоритм, который правильно обрабатывает составные слова.
    """
    
    # Если тексты идентичны, возвращаем пустой список ошибок
    if original_text == corrected_text:
        return {
            "errors": [],
            "correctedText": corrected_text
        }
    
    errors = []
    
    # Разбиваем тексты на слова с сохранением пробелов и знаков препинания
    def tokenize_with_positions(text):
        # Используем более сложное регулярное выражение для сохранения структуры текста
        pattern = r'(\S+\s*)'
        tokens = []
        pos = 0
        for match in re.finditer(pattern, text):
            token_text = match.group()
            start = match.start()
            end = match.end()
            tokens.append({
                'text': token_text,
                'start': start,
                'end': end,
                'clean_text': token_text.strip()  # Текст без пробелов для сравнения
            })
            pos = end
        return tokens
    
    original_tokens = tokenize_with_positions(original_text)
    corrected_tokens = tokenize_with_positions(corrected_text)
    
    # Преобразуем в списки чистых текстов для difflib
    original_token_texts = [t['clean_text'] for t in original_tokens]
    corrected_token_texts = [t['clean_text'] for t in corrected_tokens]
    
    # Используем difflib для сравнения токенов
    matcher = difflib.SequenceMatcher(None, original_token_texts, corrected_token_texts)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # Замена одного или нескольких токенов
            original_segment = original_tokens[i1:i2]
            corrected_segment = corrected_tokens[j1:j2]
            
            # Определяем позицию в исходном тексте
            start = original_tokens[i1]['start']
            end = original_tokens[i2-1]['end']
            
            original_phrase = ''.join([t['text'] for t in original_segment]).strip()
            corrected_phrase = ''.join([t['text'] for t in corrected_segment]).strip()
            
            # Проверяем, является ли это случаем составного слова
            is_composite_word = (
                len(original_segment) > 1 and 
                len(corrected_segment) == 1 and
                '-' in corrected_phrase
            )
            
            if is_composite_word:
                # Это составное слово - обрабатываем как одну ошибку
                errors.append({
                    "word": original_phrase,
                    "position": {
                        "start": start,
                        "end": end
                    },
                    "suggestions": [corrected_phrase],
                    "severity": "error"
                })
            else:
                # Обычная замена
                errors.append({
                    "word": original_phrase,
                    "position": {
                        "start": start,
                        "end": end
                    },
                    "suggestions": [corrected_phrase],
                    "severity": "error"
                })
        
        elif tag == 'delete':
            # Удаление токенов
            start = original_tokens[i1]['start']
            end = original_tokens[i2-1]['end']
            
            deleted_phrase = ''.join([t['text'] for t in original_tokens[i1:i2]]).strip()
            
            errors.append({
                "word": deleted_phrase,
                "position": {
                    "start": start,
                    "end": end
                },
                "suggestions": [""],
                "severity": "error"
            })
        
        elif tag == 'insert':
            # Вставка токенов
            if i1 < len(original_tokens):
                start = original_tokens[i1]['start']
            else:
                start = len(original_text)
            
            inserted_phrase = ''.join([t['text'] for t in corrected_tokens[j1:j2]]).strip()
            
            # Пропускаем вставки, которые являются частью составных слов
            # (они уже обработаны в блоке 'replace')
            is_part_of_composite = any(
                error['position']['start'] <= start <= error['position']['end'] 
                for error in errors
            )
            
            if not is_part_of_composite and inserted_phrase:
                errors.append({
                    "word": "",
                    "position": {
                        "start": start,
                        "end": start
                    },
                    "suggestions": [inserted_phrase],
                    "severity": "error"
                })
    
    # Удаляем дубликаты и сортируем
    unique_errors = []
    seen_starts = set()
    
    for error in errors:
        if error['position']['start'] not in seen_starts:
            unique_errors.append(error)
            seen_starts.add(error['position']['start'])
    
    unique_errors.sort(key=lambda x: x['position']['start'])
    
    return {
        "errors": unique_errors,
        "correctedText": corrected_text
    }

def find_text_errors_optimized(original_text: str, corrected_text: str) -> Dict[str, Any]:
    """
    Оптимизированный алгоритм с улучшенной обработкой всех типов ошибок.
    """
    
    # Если тексты идентичны, возвращаем пустой список ошибок
    if original_text == corrected_text:
        return {
            "errors": [],
            "correctedText": corrected_text
        }
    
    errors = []
    
    # Используем difflib для сравнения на уровне символов
    matcher = difflib.SequenceMatcher(None, original_text, corrected_text)
    
    # Собираем все изменения
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            changes.append((tag, i1, i2, j1, j2))
    
    # Группируем близкие изменения
    grouped_changes = []
    i = 0
    while i < len(changes):
        current_tag, current_i1, current_i2, current_j1, current_j2 = changes[i]
        
        # Ищем следующие изменения, которые близки к текущему
        group = [(current_tag, current_i1, current_i2, current_j1, current_j2)]
        j = i + 1
        while j < len(changes):
            next_tag, next_i1, next_i2, next_j1, next_j2 = changes[j]
            
            # Если изменения близки (в пределах 10 символов), группируем их
            if next_i1 - current_i2 <= 10:
                group.append((next_tag, next_i1, next_i2, next_j1, next_j2))
                current_i2 = next_i2
                j += 1
            else:
                break
        
        # Объединяем группу изменений в одну ошибку
        if len(group) == 1:
            # Одиночное изменение
            tag, i1, i2, j1, j2 = group[0]
            
            # Определяем границы слова в оригинальном тексте
            start = i1
            end = i2
            
            # Расширяем границы до целых слов
            while start > 0 and original_text[start-1].isalnum():
                start -= 1
            while end < len(original_text) and original_text[end].isalnum():
                end += 1
            
            original_word = original_text[start:end]
            
            # Определяем границы слова в исправленном тексте
            corr_start = j1
            corr_end = j2
            while corr_start > 0 and corrected_text[corr_start-1].isalnum():
                corr_start -= 1
            while corr_end < len(corrected_text) and corrected_text[corr_end].isalnum():
                corr_end += 1
            
            corrected_word = corrected_text[corr_start:corr_end]
            
            errors.append({
                "word": original_word,
                "position": {
                    "start": start,
                    "end": end
                },
                "suggestions": [corrected_word],
                "severity": "error"
            })
        else:
            # Группа изменений - вероятно, составное слово
            start = group[0][1]
            end = group[-1][2]
            
            # Определяем исправленный текст для этой группы
            corr_start = group[0][3]
            corr_end = group[-1][4]
            corrected_segment = corrected_text[corr_start:corr_end]
            
            original_segment = original_text[start:end]
            
            errors.append({
                "word": original_segment,
                "position": {
                    "start": start,
                    "end": end
                },
                "suggestions": [corrected_segment],
                "severity": "error"
            })
        
        i = j
    
    # Сортируем ошибки по позиции
    errors.sort(key=lambda x: x['position']['start'])
    
    return {
        "errors": errors,
        "correctedText": corrected_text
    }


@app.get("/")
def root():
    model_info = {
        "type": "trained" if USE_TRAINED_MODEL else "base",
        "path": str(local_model_path) if USE_LOCAL_MODEL else "ai-forever/RuM2M100-1.2B",
        "source": "local" if USE_LOCAL_MODEL else "huggingface",
        "model_name": "RuM2M100-1.2B"
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
        
        import json
        # Находим все ошибки (орфография + пунктуация) через сравнение с исправленным текстом
        # all_errors = json.dumps(find_text_errors_optimized(original_text, corrected_text), ensure_ascii=False, indent=2)
        errors_result = find_text_errors_optimized(original_text, corrected_text)
        # Извлекаем список ошибок из словаря
        all_errors = errors_result.get("errors", [])
        # и правильно обрабатывает пробелы и пунктуацию
        final_corrected_text = corrected_text
        
        
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
    if USE_LOCAL_MODEL:
        print(f"   - Локальная модель: {local_model_path}")
    else:
        print("   - Модель из HuggingFace Hub: ai-forever/RuM2M100-1.2B")
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

