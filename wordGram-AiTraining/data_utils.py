"""
Утилиты для работы с данными для обучения модели исправления грамматики
"""
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Загружает данные из JSONL файла
    
    Args:
        file_path: Путь к JSONL файлу
        
    Returns:
        Список словарей с данными
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    Сохраняет данные в JSONL файл
    
    Args:
        data: Список словарей для сохранения
        file_path: Путь к файлу
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def prepare_dataset_from_pairs(
    incorrect_texts: List[str],
    correct_texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Подготавливает датасет из пар текстов (неправильный, правильный)
    
    Args:
        incorrect_texts: Список текстов с ошибками
        correct_texts: Список исправленных текстов
        train_ratio: Доля данных для обучения
        val_ratio: Доля данных для валидации
        seed: Seed для случайного перемешивания
        
    Returns:
        Кортеж (train_data, val_data, test_data)
    """
    assert len(incorrect_texts) == len(correct_texts), \
        "Количество неправильных и правильных текстов должно совпадать"
    
    # Создаем пары
    pairs = list(zip(incorrect_texts, correct_texts))
    
    # Перемешиваем
    random.seed(seed)
    random.shuffle(pairs)
    
    # Разделяем на train/val/test
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    
    # Преобразуем в формат для обучения
    train_data = [{"input": incorrect, "target": correct} 
                  for incorrect, correct in train_pairs]
    val_data = [{"input": incorrect, "target": correct} 
                for incorrect, correct in val_pairs]
    test_data = [{"input": incorrect, "target": correct} 
                 for incorrect, correct in test_pairs]
    
    return train_data, val_data, test_data


def create_hf_dataset(
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer: AutoTokenizer,
    config,
    test_data: Optional[List[Dict]] = None
) -> DatasetDict:
    """
    Создает HuggingFace Dataset из данных
    
    Args:
        train_data: Обучающие данные
        val_data: Валидационные данные
        tokenizer: Токенизатор модели
        config: Конфигурация обучения
        test_data: Тестовые данные (опционально)
        
    Returns:
        DatasetDict с train, validation и test датасетами
    """
    
    def preprocess_function(examples):
        """Предобработка данных для модели"""
        inputs = [config.prefix + inp for inp in examples["input"]]
        targets = examples["target"]
        
        # Токенизация входных данных
        model_inputs = tokenizer(
            inputs,
            max_length=config.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        
        # Токенизация целевых данных
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=config.max_seq_length,
                truncation=True,
                padding="max_length",
            )
        
        # Для T5 моделей labels должны быть input_ids
        model_inputs["labels"] = labels["input_ids"]
        
        # Заменяем padding токены в labels на -100 (игнорируются при вычислении loss)
        model_inputs["labels"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in model_inputs["labels"]
        ]
        
        return model_inputs
    
    # Создаем датасеты
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Применяем предобработку
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    datasets = {"train": train_dataset, "validation": val_dataset}
    
    # Добавляем тестовый датасет, если он есть
    if test_data:
        test_dataset = Dataset.from_list(test_data)
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
        )
        datasets["test"] = test_dataset
    
    return DatasetDict(datasets)


def load_russian_gec_dataset(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Загружает Russian Grammar Error Correction Dataset из CSV файла
    
    Args:
        csv_path: Путь к CSV файлу датасета
        
    Returns:
        Кортеж (incorrect_texts, correct_texts)
    """
    print(f"Загрузка датасета из {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Проверяем наличие нужных столбцов
    if 'incorrect' in df.columns and 'correct' in df.columns:
        incorrect = df['incorrect'].astype(str).tolist()
        correct = df['correct'].astype(str).tolist()
    elif 'input_text' in df.columns and 'target_text' in df.columns:
        # Используем input_text и target_text, убирая префикс если есть
        incorrect = df['input_text'].astype(str).tolist()
        correct = df['target_text'].astype(str).tolist()
        # Убираем префикс из input_text, если он есть
        prefix = "Исправить грамматику и пунктуацию: "
        incorrect = [text.replace(prefix, "").strip() if prefix in text else text 
                    for text in incorrect]
    else:
        raise ValueError(
            "CSV файл должен содержать столбцы 'incorrect'/'correct' "
            "или 'input_text'/'target_text'"
        )
    
    # Фильтруем пустые значения
    pairs = [(inc, cor) for inc, cor in zip(incorrect, correct) 
             if pd.notna(inc) and pd.notna(cor) and str(inc).strip() and str(cor).strip()]
    
    incorrect = [p[0] for p in pairs]
    correct = [p[1] for p in pairs]
    
    print(f"Загружено {len(incorrect)} пар предложений")
    
    return incorrect, correct


def load_spellcheck_datasets(spellcheck_dir: str = "data/spellCheck") -> Tuple[List[str], List[str]]:
    """
    Загружает данные из всех JSON файлов в папке spellCheck
    
    Формат данных: каждая строка в JSON файле - это JSON объект с полями:
    - source: неправильный текст
    - correction: правильный текст
    - domain: источник данных (опционально)
    
    Args:
        spellcheck_dir: Путь к директории с датасетами spellCheck
        
    Returns:
        Кортеж (incorrect_texts, correct_texts)
    """
    spellcheck_path = Path(spellcheck_dir)
    if not spellcheck_path.exists():
        print(f"[WARNING] Папка {spellcheck_dir} не найдена")
        return [], []
    
    incorrect_texts = []
    correct_texts = []
    
    # Ищем все JSON файлы в подпапках
    json_files = list(spellcheck_path.rglob("*.json"))
    
    if not json_files:
        print(f"[WARNING] JSON файлы не найдены в {spellcheck_dir}")
        return [], []
    
    print(f"Найдено {len(json_files)} JSON файлов в {spellcheck_dir}")
    
    for json_file in json_files:
        try:
            print(f"  Загрузка {json_file.name}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            file_incorrect = []
            file_correct = []
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    # Проверяем формат данных
                    if 'source' in data and 'correction' in data:
                        source = str(data['source']).strip()
                        correction = str(data['correction']).strip()
                        
                        # Фильтруем пустые значения
                        if source and correction and source != correction:
                            file_incorrect.append(source)
                            file_correct.append(correction)
                    else:
                        print(f"    [WARNING] Строка {line_num}: отсутствуют поля 'source' или 'correction'")
                except json.JSONDecodeError as e:
                    print(f"    [WARNING] Ошибка парсинга JSON на строке {line_num}: {e}")
                    continue
            
            incorrect_texts.extend(file_incorrect)
            correct_texts.extend(file_correct)
            print(f"    [OK] Загружено {len(file_incorrect)} пар из {json_file.name}")
            
        except Exception as e:
            print(f"    [ERROR] Ошибка при загрузке {json_file.name}: {e}")
            continue
    
    print(f"\n[OK] Всего загружено {len(incorrect_texts)} пар из spellCheck датасетов")
    
    return incorrect_texts, correct_texts


def load_example_dataset() -> Tuple[List[str], List[str]]:
    """
    Создает пример датасета для демонстрации
    Можно заменить на загрузку реальных данных
    
    Returns:
        Кортеж (incorrect_texts, correct_texts)
    """
    # Примеры текстов с ошибками и их исправленные версии
    examples = [
        ("привет как дела", "Привет, как дела?"),
        ("я иду в магазин купить хлеб", "Я иду в магазин купить хлеб."),
        ("вчера я был в кинотеатре это было круто", "Вчера я был в кинотеатре. Это было круто!"),
        ("она не знает что делать дальше", "Она не знает, что делать дальше."),
        ("мы пошли гулять но пошел дождь", "Мы пошли гулять, но пошел дождь."),
        ("он сказал что придет завтра", "Он сказал, что придет завтра."),
        ("она купила яблоки груши и бананы", "Она купила яблоки, груши и бананы."),
        ("когда приедешь позвони мне", "Когда приедешь, позвони мне."),
        ("я хочу поесть но не знаю что", "Я хочу поесть, но не знаю что."),
        ("он работает много и устал", "Он работает много и устал."),
    ]
    
    incorrect = [ex[0] for ex in examples]
    correct = [ex[1] for ex in examples]
    
    return incorrect, correct
