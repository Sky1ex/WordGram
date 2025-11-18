"""
Скрипт для подготовки данных для обучения модели
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from data_utils import (
    save_jsonl,
    prepare_dataset_from_pairs,
    load_example_dataset,
    load_jsonl,
    load_russian_gec_dataset,
    load_spellcheck_datasets
)
from config import config


def create_example_data(output_dir: str = "data"):
    """
    Создает пример датасета для демонстрации
    
    Args:
        output_dir: Директория для сохранения данных
    """
    print("Создание примера датасета...")
    
    # Загружаем примеры
    incorrect, correct = load_example_dataset()
    
    # Подготавливаем датасет
    train_data, val_data, test_data = prepare_dataset_from_pairs(
        incorrect_texts=incorrect,
        correct_texts=correct,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    
    # Сохраняем
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(val_data, f"{output_dir}/val.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")
    
    print(f"[OK] Данные сохранены в {output_dir}/")
    print(f"  Обучающих примеров: {len(train_data)}")
    print(f"  Валидационных примеров: {len(val_data)}")
    print(f"  Тестовых примеров: {len(test_data)}")


def prepare_data_from_file(input_file: str, output_dir: str = "data", format: str = "jsonl"):
    """
    Подготавливает данные из файла
    
    Args:
        input_file: Путь к входному файлу
        output_dir: Директория для сохранения данных
        format: Формат входного файла (jsonl, json, csv)
    """
    print(f"Загрузка данных из {input_file}...")
    
    # Специальная обработка для Russian GEC Dataset
    if format == "csv" and Path(input_file).exists():
        # Проверяем, является ли это Russian GEC Dataset
        try:
            df = pd.read_csv(input_file, encoding='utf-8', nrows=1)
            if 'incorrect' in df.columns or 'input_text' in df.columns:
                # Используем специальную функцию для загрузки
                incorrect, correct = load_russian_gec_dataset(input_file)
                train_data, val_data, test_data = prepare_dataset_from_pairs(
                    incorrect_texts=incorrect,
                    correct_texts=correct,
                    train_ratio=0.8,
                    val_ratio=0.1,
                    seed=42
                )
                
                # Сохраняем
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                save_jsonl(train_data, f"{output_dir}/train.jsonl")
                save_jsonl(val_data, f"{output_dir}/val.jsonl")
                save_jsonl(test_data, f"{output_dir}/test.jsonl")
                
                print(f"[OK] Данные сохранены в {output_dir}/")
                print(f"  Обучающих примеров: {len(train_data)}")
                print(f"  Валидационных примеров: {len(val_data)}")
                print(f"  Тестовых примеров: {len(test_data)}")
                return
        except Exception as e:
            print(f"Предупреждение: не удалось загрузить как Russian GEC Dataset: {e}")
            print("Попытка загрузить как обычный CSV...")
    
    data = []
    
    if format == "jsonl":
        data = load_jsonl(input_file)
    elif format == "json":
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Если это словарь с ключами "input" и "target"
                if "input" in data and "target" in data:
                    data = [{"input": inp, "target": tar} 
                           for inp, tar in zip(data["input"], data["target"])]
    elif format == "csv":
        # Загружаем CSV как обычный файл
        df = pd.read_csv(input_file, encoding='utf-8')
        # Пытаемся найти нужные столбцы
        if 'input' in df.columns and 'target' in df.columns:
            data = [{"input": str(row['input']), "target": str(row['target'])} 
                   for _, row in df.iterrows()]
        elif 'incorrect' in df.columns and 'correct' in df.columns:
            data = [{"input": str(row['incorrect']), "target": str(row['correct'])} 
                   for _, row in df.iterrows()]
        else:
            # Если не нашли нужные столбцы, используем первые два
            cols = df.columns.tolist()
            if len(cols) >= 2:
                data = [{"input": str(row[cols[0]]), "target": str(row[cols[1]])} 
                       for _, row in df.iterrows()]
            else:
                raise ValueError(f"CSV файл должен содержать как минимум 2 столбца. Найдено: {cols}")
    
    if not data:
        raise ValueError(f"Не удалось загрузить данные из {input_file}")
    
    print(f"Загружено {len(data)} примеров")
    
    # Проверяем формат данных
    if not all("input" in item and "target" in item for item in data):
        raise ValueError("Данные должны содержать поля 'input' и 'target'")
    
    # Разделяем на train/val/test
    incorrect = [item["input"] for item in data]
    correct = [item["target"] for item in data]
    
    train_data, val_data, test_data = prepare_dataset_from_pairs(
        incorrect_texts=incorrect,
        correct_texts=correct,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    
    # Сохраняем
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(val_data, f"{output_dir}/val.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")
    
    print(f"[OK] Данные сохранены в {output_dir}/")
    print(f"  Обучающих примеров: {len(train_data)}")
    print(f"  Валидационных примеров: {len(val_data)}")
    print(f"  Тестовых примеров: {len(test_data)}")


def prepare_data_from_spellcheck(spellcheck_dir: str = "data/spellCheck", output_dir: str = "data"):
    """
    Подготавливает данные из папки spellCheck
    
    Args:
        spellcheck_dir: Путь к директории с датасетами spellCheck
        output_dir: Директория для сохранения данных
    """
    print("Загрузка данных из spellCheck датасетов...")
    
    # Загружаем данные из spellCheck
    incorrect, correct = load_spellcheck_datasets(spellcheck_dir)
    
    if not incorrect or not correct:
        print("[ERROR] Не удалось загрузить данные из spellCheck")
        return
    
    # Подготавливаем датасет
    train_data, val_data, test_data = prepare_dataset_from_pairs(
        incorrect_texts=incorrect,
        correct_texts=correct,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    
    # Сохраняем
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(val_data, f"{output_dir}/val.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")
    
    print(f"\n[OK] Данные сохранены в {output_dir}/")
    print(f"  Обучающих примеров: {len(train_data)}")
    print(f"  Валидационных примеров: {len(val_data)}")
    print(f"  Тестовых примеров: {len(test_data)}")


def prepare_data_combined(
    input_file: str = None,
    spellcheck_dir: str = "data/spellCheck",
    output_dir: str = "data",
    format: str = "jsonl"
):
    """
    Подготавливает данные, объединяя данные из файла и из spellCheck
    
    Args:
        input_file: Путь к входному файлу с данными (опционально)
        spellcheck_dir: Путь к директории с датасетами spellCheck
        output_dir: Директория для сохранения данных
        format: Формат входного файла
    """
    print("=" * 60)
    print("Подготовка комбинированного датасета")
    print("=" * 60)
    
    all_incorrect = []
    all_correct = []
    
    # Загружаем данные из spellCheck
    if Path(spellcheck_dir).exists():
        print("\n[1/2] Загрузка данных из spellCheck...")
        incorrect_sc, correct_sc = load_spellcheck_datasets(spellcheck_dir)
        all_incorrect.extend(incorrect_sc)
        all_correct.extend(correct_sc)
        print(f"  Загружено из spellCheck: {len(incorrect_sc)} пар")
    else:
        print(f"\n[1/2] [WARNING] Папка {spellcheck_dir} не найдена, пропускаем")
    
    # Загружаем данные из файла, если указан
    if input_file and Path(input_file).exists():
        print(f"\n[2/2] Загрузка данных из файла {input_file}...")
        try:
            if format == "csv":
                incorrect_file, correct_file = load_russian_gec_dataset(input_file)
            else:
                # Загружаем как обычный файл
                data = load_jsonl(input_file) if format == "jsonl" else []
                if data:
                    incorrect_file = [item.get("input", "") for item in data]
                    correct_file = [item.get("target", "") for item in data]
                else:
                    incorrect_file, correct_file = [], []
            
            all_incorrect.extend(incorrect_file)
            all_correct.extend(correct_file)
            print(f"  Загружено из файла: {len(incorrect_file)} пар")
        except Exception as e:
            print(f"  [WARNING] Ошибка при загрузке файла: {e}")
    elif input_file:
        print(f"\n[2/2] [WARNING] Файл {input_file} не найден, пропускаем")
    else:
        print(f"\n[2/2] Файл не указан, пропускаем")
    
    if not all_incorrect or not all_correct:
        print("\n[ERROR] Не удалось загрузить данные из указанных источников")
        return
    
    print(f"\n{'=' * 60}")
    print(f"Всего загружено: {len(all_incorrect)} пар")
    print(f"{'=' * 60}")
    
    # Подготавливаем датасет
    train_data, val_data, test_data = prepare_dataset_from_pairs(
        incorrect_texts=all_incorrect,
        correct_texts=all_correct,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )
    
    # Сохраняем
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_jsonl(train_data, f"{output_dir}/train.jsonl")
    save_jsonl(val_data, f"{output_dir}/val.jsonl")
    save_jsonl(test_data, f"{output_dir}/test.jsonl")
    
    print(f"\n[OK] Данные сохранены в {output_dir}/")
    print(f"  Обучающих примеров: {len(train_data)}")
    print(f"  Валидационных примеров: {len(val_data)}")
    print(f"  Тестовых примеров: {len(test_data)}")


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Подготовка данных для обучения модели исправления грамматики"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Путь к входному файлу с данными (опционально)"
    )
    parser.add_argument(
        "--spellcheck",
        type=str,
        default="data/spellCheck",
        help="Путь к директории с датасетами spellCheck (по умолчанию: data/spellCheck)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Директория для сохранения подготовленных данных (по умолчанию: data)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "csv"],
        help="Формат входного файла (по умолчанию: jsonl)"
    )
    parser.add_argument(
        "--spellcheck-only",
        action="store_true",
        help="Использовать только данные из spellCheck (игнорировать --input)"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Объединить данные из файла и из spellCheck (по умолчанию: True, если указаны оба)"
    )
    
    args = parser.parse_args()
    
    # Если указан только spellcheck-only, используем только spellCheck
    if args.spellcheck_only:
        prepare_data_from_spellcheck(args.spellcheck, args.output)
    # Если указан input и spellcheck существует, объединяем по умолчанию (если не указано иное)
    elif args.input and Path(args.spellcheck).exists():
        # По умолчанию объединяем, если не указано явно использовать только файл
        prepare_data_combined(args.input, args.spellcheck, args.output, args.format)
    # Если указан только input (spellcheck не существует или не указан)
    elif args.input:
        prepare_data_from_file(args.input, args.output, args.format)
    # Если указан только spellcheck (без input) - используем по умолчанию
    elif Path(args.spellcheck).exists():
        prepare_data_from_spellcheck(args.spellcheck, args.output)
    # По умолчанию создаем пример
    else:
        print("[WARNING] Не указаны источники данных. Создаем пример датасета...")
        print("   Используйте --spellcheck для загрузки из spellCheck")
        print("   или --input для загрузки из файла")
        create_example_data(args.output)


if __name__ == "__main__":
    main()
