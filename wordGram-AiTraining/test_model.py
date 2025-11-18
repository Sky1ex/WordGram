"""
Скрипт для быстрого тестирования обученной модели
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
from config import config


def test_model(model_path: str, texts: list = None):
    """
    Тестирует обученную модель на примерах
    
    Args:
        model_path: Путь к обученной модели
        texts: Список текстов для тестирования
    """
    if texts is None:
        texts = [
            "привет как дела",
            "я иду в магазин купить хлеб",
            "она не знает что делать дальше",
            "мы пошли гулять но пошел дождь",
            "он сказал что придет завтра",
        ]
    
    print("=" * 60)
    print("Тестирование модели исправления грамматики")
    print("=" * 60)
    
    # Проверяем наличие модели
    if not Path(model_path).exists():
        print(f"❌ Ошибка: Модель не найдена в {model_path}")
        print("   Сначала обучите модель с помощью train.py")
        return
    
    # Проверяем наличие GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"\nИспользуемое устройство: {'GPU' if device >= 0 else 'CPU'}")
    
    # Загружаем модель
    print(f"\nЗагрузка модели из {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        # Создаем pipeline
        corrector = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("✓ Модель загружена успешно")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return
    
    # Тестируем на примерах
    print("\n" + "-" * 60)
    print("Результаты исправления:")
    print("-" * 60)
    
    prefix = config.prefix if hasattr(config, 'prefix') else "Исправить грамматику и пунктуацию: "
    
    for i, text in enumerate(texts, 1):
        try:
            result = corrector(
                prefix + text,
                max_length=config.max_length if hasattr(config, 'max_length') else 512,
                num_beams=config.num_beams if hasattr(config, 'num_beams') else 5,
                early_stopping=True,
                do_sample=False
            )
            
            corrected = result[0]["generated_text"].strip()
            # Убираем префикс, если он есть
            if prefix in corrected:
                corrected = corrected.replace(prefix, "").strip()
            
            print(f"\n{i}. Исходный текст:")
            print(f"   {text}")
            print(f"   Исправленный:")
            print(f"   {corrected}")
        except Exception as e:
            print(f"\n{i}. Ошибка при обработке текста '{text}': {e}")
    
    print("\n" + "=" * 60)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Тестирование обученной модели исправления грамматики"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.output_dir,
        help=f"Путь к обученной модели (по умолчанию: {config.output_dir})"
    )
    parser.add_argument(
        "--text",
        type=str,
        action="append",
        help="Текст для тестирования (можно указать несколько раз)"
    )
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        texts=args.text if args.text else None
    )


if __name__ == "__main__":
    main()
