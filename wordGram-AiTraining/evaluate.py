"""
Скрипт для оценки обученной модели
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import config
from data_utils import load_jsonl
import argparse
from pathlib import Path


def evaluate_model(
    model_path: str,
    test_data_path: str = None,
    examples: list = None
):
    """
    Оценивает обученную модель
    
    Args:
        model_path: Путь к обученной модели
        test_data_path: Путь к тестовым данным (опционально)
        examples: Список примеров для тестирования (опционально)
    """
    print("=" * 60)
    print("Оценка модели исправления грамматики")
    print("=" * 60)
    
    # Проверяем наличие GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"\nИспользуемое устройство: {'GPU' if device >= 0 else 'CPU'}")
    
    # Загружаем модель и токенизатор
    print(f"\n[1/3] Загрузка модели из {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Создаем pipeline для генерации
    corrector = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    print("✓ Модель загружена")
    
    # Подготавливаем примеры для оценки
    if test_data_path and Path(test_data_path).exists():
        print(f"\n[2/3] Загрузка тестовых данных из {test_data_path}...")
        test_data = load_jsonl(test_data_path)
        examples = [
            {"input": item["input"], "target": item["target"]}
            for item in test_data[:50]  # Ограничиваем до 50 примеров для быстрой оценки
        ]
        print(f"✓ Загружено {len(examples)} примеров")
    elif examples is None:
        # Используем примеры по умолчанию
        examples = [
            {"input": "привет как дела", "target": "Привет, как дела?"},
            {"input": "я иду в магазин купить хлеб", "target": "Я иду в магазин купить хлеб."},
            {"input": "она не знает что делать дальше", "target": "Она не знает, что делать дальше."},
            {"input": "мы пошли гулять но пошел дождь", "target": "Мы пошли гулять, но пошел дождь."},
            {"input": "он сказал что придет завтра", "target": "Он сказал, что придет завтра."},
        ]
        print(f"\n[2/3] Использование примеров по умолчанию ({len(examples)} примеров)")
    else:
        print(f"\n[2/3] Использование предоставленных примеров ({len(examples)} примеров)")
    
    # Оцениваем модель
    print(f"\n[3/3] Оценка модели...")
    print("\n" + "-" * 60)
    print("Результаты исправления:")
    print("-" * 60)
    
    correct_count = 0
    total_count = len(examples)
    
    for i, example in enumerate(examples, 1):
        input_text = example["input"]
        target_text = example["target"]
        
        # Генерируем исправление
        prefix = config.prefix if hasattr(config, 'prefix') else "Исправить грамматику и пунктуацию: "
        result = corrector(
            prefix + input_text,
            max_length=config.max_length,
            num_beams=config.num_beams,
            early_stopping=config.early_stopping,
            do_sample=False
        )
        
        predicted_text = result[0]["generated_text"].strip()
        
        # Убираем префикс, если он есть в результате
        if prefix in predicted_text:
            predicted_text = predicted_text.replace(prefix, "").strip()
        
        # Проверяем точное совпадение
        is_correct = predicted_text.lower().strip() == target_text.lower().strip()
        if is_correct:
            correct_count += 1
        
        # Выводим результат
        status = "✓" if is_correct else "✗"
        print(f"\nПример {i} [{status}]:")
        print(f"  Исходный:    {input_text}")
        print(f"  Ожидаемый:   {target_text}")
        print(f"  Полученный:  {predicted_text}")
    
    # Выводим статистику
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "=" * 60)
    print("Статистика:")
    print("=" * 60)
    print(f"Всего примеров: {total_count}")
    print(f"Правильных: {correct_count}")
    print(f"Точность (exact match): {accuracy:.2%}")
    print("=" * 60)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Оценка обученной модели исправления грамматики"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.output_dir,
        help=f"Путь к обученной модели (по умолчанию: {config.output_dir})"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=config.test_data_path,
        help="Путь к тестовым данным (опционально)"
    )
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ Ошибка: Модель не найдена в {args.model}")
        print("   Сначала обучите модель с помощью train.py")
        return
    
    evaluate_model(
        model_path=args.model,
        test_data_path=args.test_data
    )


if __name__ == "__main__":
    main()
