import difflib
import re
from typing import List, Dict, Any

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

# Тестирование
if __name__ == "__main__":
    original_text = "Основая цель мероприятия - практическая отработка навыков по оказанию помощи гражданам, попавшим в ДТП, а также повышение и совершенствование уровня профессиональной подготовки сотрудников МЧС при проведении аварийно спасательных работ по ликвидации последствий дорожно транспортных проишествий, сокращение временных показателей реагирования."
    
    corrected_text = "Основная цель мероприятия - практическая отработка навыков по оказанию помощи гражданам, попавшим в ДТП, а также повышение и совершенствование уровня профессиональной подготовки сотрудников МЧС при проведении аварийно-спасательных работ по ликвидации последствий дорожно-транспортных происшествий, сокращение временных показателей реагирования."
    
    print("=== Исправленный алгоритм ===")
    result1 = find_text_errors_corrected(original_text, corrected_text)
    
    print("=== Оптимизированный алгоритм ===")
    result2 = find_text_errors_optimized(original_text, corrected_text)
    
    import json
    print("Результат оптимизированного алгоритма:")
    print(json.dumps(result2, ensure_ascii=False, indent=2))