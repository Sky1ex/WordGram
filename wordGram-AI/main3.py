from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
        
        # Очистка и нормализация результата
        corrected_text = corrected_text.strip()
        
        # Нормализуем пробелы
        corrected_text = re.sub(r'\s+', ' ', corrected_text)
        
        # Убираем множественные знаки препинания
        corrected_text = re.sub(r'([.,!?:;])\1+', r'\1', corrected_text)
        
        # Убираем знаки препинания в начале
        corrected_text = re.sub(r'^[.,!?:;]+', '', corrected_text)
        
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
        
        # Строим маппинг позиций от оригинала к исправленному тексту
        # Это позволяет точно находить соответствующие сегменты
        def build_position_mapping(orig: str, corr: str):
            """Строит маппинг позиций между оригиналом и исправленным текстом"""
            matcher = difflib.SequenceMatcher(None, orig, corr, autojunk=False)
            mapping = {}  # orig_pos -> corr_pos
            
            orig_pos = 0
            corr_pos = 0
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                # Маппинг для равных сегментов
                if tag == 'equal':
                    for k in range(i2 - i1):
                        mapping[i1 + k] = j1 + k
                    orig_pos = i2
                    corr_pos = j2
                # Для замен - маппим начальную позицию
                elif tag == 'replace':
                    mapping[i1] = j1
                    orig_pos = i2
                    corr_pos = j2
                # Для вставок - маппим позицию перед вставкой
                elif tag == 'insert':
                    if i1 > 0:
                        mapping[i1] = j1
                    orig_pos = i1
                    corr_pos = j2
                # Для удалений - маппим позицию перед удалением
                elif tag == 'delete':
                    mapping[i1] = j1 if j1 > 0 else 0
                    orig_pos = i2
                    corr_pos = j1
            
            return mapping
        
        mapping = build_position_mapping(original, corrected_text)
        
        # Используем difflib для точного сравнения на уровне символов
        matcher = difflib.SequenceMatcher(None, original, corrected_text, autojunk=False)
        found_positions = set()
        
        # Собираем все различия, группируя соседние изменения
        diff_blocks = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                diff_blocks.append((tag, i1, i2, j1, j2))
        
        # Обрабатываем каждый блок различий
        for tag, i1, i2, j1, j2 in diff_blocks:
            if tag == 'replace':
                # Замена: находим границы слова/фразы в оригинале
                start = i1
                end = i2
                
                # Расширяем назад до границы слова (пробел или знак препинания)
                while start > 0 and original[start - 1] not in ' \n\t.,!?:;…':
                    start -= 1
                
                # Расширяем вперед до границы слова
                while end < len(original) and original[end] not in ' \n\t.,!?:;…':
                    end += 1
                
                # Извлекаем исходное слово/фразу
                orig_segment = original[start:end]
                
                # Находим соответствующий сегмент в исправленном тексте используя контекст
                # Берем достаточно контекста до и после изменения
                context_before = original[max(0, start - 15):start]
                context_after = original[end:min(len(original), end + 15)]
                
                # Ищем контекст в исправленном тексте
                # Ищем конец контекста "до" в исправленном тексте
                search_start = max(0, j1 - 30)
                corr_context_before_end = corrected_text.find(context_before[-min(10, len(context_before)):], search_start, j2 + 30)
                
                if corr_context_before_end != -1:
                    corr_start = corr_context_before_end + len(context_before[-min(10, len(context_before)):])
                    # Ищем начало контекста "после"
                    corr_context_after_start = corrected_text.find(context_after[:min(10, len(context_after))], corr_start, min(len(corrected_text), corr_start + 50))
                    if corr_context_after_start != -1:
                        corr_end = corr_context_after_start
                    else:
                        # Если не нашли контекст после, расширяем до границы слова
                        corr_end = j2
                        while corr_end < len(corrected_text) and corrected_text[corr_end] not in ' \n\t.,!?:;…':
                            corr_end += 1
                else:
                    # Если не нашли контекст, используем прямые позиции из difflib
                    corr_start = j1
                    corr_end = j2
                
                # Расширяем до границ слова в исправленном тексте
                while corr_start > 0 and corrected_text[corr_start - 1] not in ' \n\t.,!?:;…':
                    corr_start -= 1
                while corr_end < len(corrected_text) and corrected_text[corr_end] not in ' \n\t.,!?:;…':
                    corr_end += 1
                
                # Извлекаем исправленный сегмент
                corr_segment = corrected_text[corr_start:corr_end].strip()
                
                # Пропускаем, если сегменты одинаковые (после нормализации пробелов)
                if orig_segment.strip().replace(' ', '') == corr_segment.replace(' ', ''):
                    continue
                
                # Проверяем, не перекрывается ли с уже найденной ошибкой
                pos_key = (start, end)
                overlapping = False
                for existing_start, existing_end in found_positions:
                    if not (end <= existing_start or start >= existing_end):
                        overlapping = True
                        break
                
                if not overlapping and orig_segment.strip():
                    errors.append({
                        "word": orig_segment,
                        "position": {"start": start, "end": end},
                        "suggestions": [corr_segment] if corr_segment else [""],
                        "severity": "error"
                    })
                    found_positions.add(pos_key)
            
            elif tag == 'insert':
                # Вставка: в исправленном тексте добавлены символы
                inserted_text = corrected_text[j1:j2]
                
                # Пропускаем вставку только пробелов (это может быть нормализация)
                # Но проверяем контекст - если это разделение слова, обрабатываем
                if inserted_text.strip() == '':
                    # Пропускаем простые пробелы
                    continue
                
                # Находим позицию вставки в оригинале
                insert_pos = i1
                
                # Находим слово перед позицией вставки в оригинале
                word_before_start = insert_pos
                while word_before_start > 0 and original[word_before_start - 1] not in ' \n\t.,!?:;…':
                    word_before_start -= 1
                
                word_before = original[word_before_start:insert_pos] if word_before_start < insert_pos else ""
                
                # Находим слово после позиции вставки в оригинале (если есть)
                word_after_end = insert_pos
                while word_after_end < len(original) and original[word_after_end] not in ' \n\t.,!?:;…':
                    word_after_end += 1
                
                word_after = original[insert_pos:word_after_end] if word_after_end > insert_pos else ""
                
                # Если вставлен только знак препинания
                if re.match(r'^[.,!?:;…]+$', inserted_text.strip()):
                    if word_before.strip():
                        pos_key = (word_before_start, insert_pos)
                        if pos_key not in found_positions:
                            errors.append({
                                "word": word_before,
                                "position": {"start": word_before_start, "end": insert_pos},
                                "suggestions": [word_before + inserted_text.strip()],
                                "severity": "error"
                            })
                            found_positions.add(pos_key)
                else:
                    # Вставлен текст (возможно разделение слова)
                    # Проверяем, может ли это быть частью слова перед или после
                    # Находим полное слово, которое могло быть разделено
                    full_word_start = word_before_start
                    full_word_end = word_after_end
                    full_word = original[full_word_start:full_word_end]
                    
                    # Находим соответствующую часть в исправленном тексте
                    # Используем контекст для точного нахождения
                    if full_word.strip():
                        # Ищем контекст вокруг этого слова
                        context_before = original[max(0, full_word_start - 10):full_word_start]
                        context_after = original[full_word_end:min(len(original), full_word_end + 10)]
                        
                        # Ищем в исправленном тексте
                        search_start = max(0, j1 - 30)
                        corr_before_pos = corrected_text.find(context_before[-min(5, len(context_before)):], search_start, j2 + 30)
                        
                        if corr_before_pos != -1:
                            corr_word_start = corr_before_pos + len(context_before[-min(5, len(context_before)):])
                            # Ищем контекст после
                            corr_after_pos = corrected_text.find(context_after[:min(5, len(context_after))], corr_word_start, min(len(corrected_text), corr_word_start + 50))
                            if corr_after_pos != -1:
                                corr_word_end = corr_after_pos
                            else:
                                corr_word_end = j2
                                while corr_word_end < len(corrected_text) and corrected_text[corr_word_end] not in ' \n\t.,!?:;…':
                                    corr_word_end += 1
                            
                            # Расширяем до границ слова
                            while corr_word_start > 0 and corrected_text[corr_word_start - 1] not in ' \n\t.,!?:;…':
                                corr_word_start -= 1
                            while corr_word_end < len(corrected_text) and corrected_text[corr_word_end] not in ' \n\t.,!?:;…':
                                corr_word_end += 1
                            
                            corr_full = corrected_text[corr_word_start:corr_word_end].strip()
                            
                            pos_key = (full_word_start, full_word_end)
                            if pos_key not in found_positions and full_word.strip() and corr_full != full_word:
                                errors.append({
                                    "word": full_word,
                                    "position": {"start": full_word_start, "end": full_word_end},
                                    "suggestions": [corr_full],
                                    "severity": "error"
                                })
                                found_positions.add(pos_key)
                                continue
                    
                    # Если не нашли как разделение слова, обрабатываем как простую вставку
                    pos_key = (insert_pos, insert_pos)
                    if pos_key not in found_positions:
                        errors.append({
                            "word": word_before if word_before.strip() else (word_after if word_after.strip() else original[max(0, insert_pos-3):insert_pos]),
                            "position": {"start": insert_pos, "end": insert_pos},
                            "suggestions": [inserted_text.strip()],
                            "severity": "error"
                        })
                        found_positions.add(pos_key)
            
            elif tag == 'delete':
                # Удаление: символы были удалены из оригинала
                deleted_text = original[i1:i2]
                
                if not deleted_text.strip():
                    continue
                
                # Находим границы слова
                start = i1
                end = i2
                
                while start > 0 and original[start - 1] not in ' \n\t.,!?:;…':
                    start -= 1
                while end < len(original) and original[end] not in ' \n\t.,!?:;…':
                    end += 1
                
                deleted_word = original[start:end]
                pos_key = (start, end)
                
                if pos_key not in found_positions:
                    errors.append({
                        "word": deleted_word,
                        "position": {"start": start, "end": end},
                        "suggestions": [""],
                        "severity": "error"
                    })
                    found_positions.add(pos_key)
        
        # Дополнительная проверка: находим все слова в оригинале и сравниваем с исправленным текстом
        # Это помогает найти ошибки, которые могли быть пропущены
        def find_word_boundaries(text: str, pos: int):
            """Находит границы слова, содержащего позицию pos"""
            start = pos
            end = pos
            while start > 0 and text[start - 1] not in ' \n\t.,!?:;…':
                start -= 1
            while end < len(text) and text[end] not in ' \n\t.,!?:;…':
                end += 1
            return start, end
        
        # Находим все слова в оригинале
        word_pattern = r'\b\w+\b'
        for match in re.finditer(word_pattern, original):
            word_start = match.start()
            word_end = match.end()
            orig_word = match.group()
            
            # Проверяем, не покрыта ли уже эта позиция
            covered = False
            for err_start, err_end in found_positions:
                if not (word_end <= err_start or word_start >= err_end):
                    covered = True
                    break
            
            if covered:
                continue
            
            # Находим соответствующее слово в исправленном тексте используя контекст
            context_before = original[max(0, word_start - 20):word_start]
            context_after = original[word_end:min(len(original), word_end + 20)]
            
            # Ищем контекст в исправленном тексте
            search_start = 0
            if context_before:
                corr_before_pos = corrected_text.find(context_before[-min(10, len(context_before)):], search_start)
                if corr_before_pos != -1:
                    corr_word_start = corr_before_pos + len(context_before[-min(10, len(context_before)):])
                    # Ищем контекст после
                    if context_after:
                        corr_after_pos = corrected_text.find(context_after[:min(10, len(context_after))], corr_word_start)
                        if corr_after_pos != -1:
                            corr_word_end = corr_after_pos
                        else:
                            # Расширяем до границы слова
                            corr_word_end = corr_word_start
                            while corr_word_end < len(corrected_text) and corrected_text[corr_word_end] not in ' \n\t.,!?:;…':
                                corr_word_end += 1
                    else:
                        corr_word_end = corr_word_start
                        while corr_word_end < len(corrected_text) and corrected_text[corr_word_end] not in ' \n\t.,!?:;…':
                            corr_word_end += 1
                    
                    # Находим все слова между контекстом до и после в исправленном тексте
                    # Это позволяет правильно обрабатывать случаи разделения слов
                    # Ищем все символы между найденными позициями
                    corr_start_expanded = corr_word_start
                    corr_end_expanded = corr_word_end
                    
                    # Расширяем назад до начала слова или пробела перед контекстом
                    while corr_start_expanded > 0 and corrected_text[corr_start_expanded - 1] not in ' \n\t':
                        if corrected_text[corr_start_expanded - 1] in '.,!?:;…':
                            break
                        corr_start_expanded -= 1
                    
                    # Расширяем вперед до конца слова или пробела после контекста
                    while corr_end_expanded < len(corrected_text) and corrected_text[corr_end_expanded] not in ' \n\t':
                        if corrected_text[corr_end_expanded] in '.,!?:;…':
                            break
                        corr_end_expanded += 1
                    
                    # Извлекаем сегмент из исправленного текста (может содержать несколько слов)
                    corr_segment = corrected_text[corr_start_expanded:corr_end_expanded].strip()
                    
                    # Сравниваем слова (нормализуем пробелы)
                    orig_normalized = orig_word
                    corr_normalized = re.sub(r'\s+', ' ', corr_segment).strip()
                    
                    # Если слова различаются, добавляем ошибку
                    if orig_normalized != corr_normalized and corr_normalized:
                        # Проверяем, что это действительно ошибка
                        # Сравниваем нормализованные версии (без пробелов и регистра)
                        orig_norm = re.sub(r'\s+', '', orig_normalized.lower())
                        corr_norm = re.sub(r'\s+', '', corr_normalized.lower())
                        
                        if orig_norm != corr_norm:
                            pos_key = (word_start, word_end)
                            if pos_key not in found_positions:
                                errors.append({
                                    "word": orig_word,
                                    "position": {"start": word_start, "end": word_end},
                                    "suggestions": [corr_normalized],
                                    "severity": "error"
                                })
                                found_positions.add(pos_key)
        
        # Сортируем ошибки по позиции
        errors.sort(key=lambda x: x['position']['start'])
        
        # Удаляем перекрывающиеся ошибки (оставляем более широкие)
        final_errors = []
        for error in errors:
            overlapping = False
            err_start = error['position']['start']
            err_end = error['position']['end']
            
            for existing in final_errors:
                ex_start = existing['position']['start']
                ex_end = existing['position']['end']
                # Проверяем пересечение
                if not (err_end <= ex_start or err_start >= ex_end):
                    # Если текущая ошибка полностью внутри существующей, пропускаем
                    if err_start >= ex_start and err_end <= ex_end:
                        overlapping = True
                        break
                    # Если существующая полностью внутри текущей, заменяем
                    elif ex_start >= err_start and ex_end <= err_end:
                        final_errors.remove(existing)
                        break
            
            if not overlapping:
                final_errors.append(error)
        
        # Снова сортируем после удаления перекрытий
        final_errors.sort(key=lambda x: x['position']['start'])
        
        return final_errors
    
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

