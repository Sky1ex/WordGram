"""
Скрипт для загрузки Russian Grammar Error Correction Dataset с GitHub
"""
import os
import urllib.request
import urllib.parse
from pathlib import Path
import argparse


def download_dataset(
    output_dir: str = "data",
    filename: str = "russian_gec_dataset_final (1).csv",
    github_url: str = "https://raw.githubusercontent.com/dreuxx/Russian-Grammar-Error-Correction-Dataset/main",
    force: bool = False
):
    """
    Загружает датасет с GitHub
    
    Args:
        output_dir: Директория для сохранения датасета
        filename: Имя файла датасета
        github_url: Базовый URL репозитория GitHub
        force: Перезаписать файл, если он уже существует
    """
    # Создаем директорию
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # URL файла (правильно экранируем для URL)
    encoded_filename = urllib.parse.quote(filename, safe='')
    file_url = f"{github_url}/{encoded_filename}"
    
    output_path = Path(output_dir) / filename
    
    # Проверяем, существует ли файл
    if output_path.exists():
        if force:
            print(f"⚠ Файл {output_path} уже существует. Перезаписываем...")
            output_path.unlink()
        else:
            print(f"⚠ Файл {output_path} уже существует.")
            print("   Используйте --force для перезаписи или удалите файл вручную.")
            return output_path
    
    print(f"Загрузка датасета с GitHub...")
    print(f"URL: {file_url}")
    print(f"Сохранение в: {output_path}")
    
    try:
        # Загружаем файл
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rПрогресс: {percent}%", end='', flush=True)
        
        urllib.request.urlretrieve(file_url, output_path, reporthook=progress_hook)
        print("\n✓ Датасет успешно загружен!")
        
        # Показываем информацию о файле
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  Размер файла: {file_size:.2f} MB")
        print(f"  Путь: {output_path}")
        
        return output_path
        
    except urllib.error.HTTPError as e:
        print(f"\n❌ Ошибка при загрузке: HTTP {e.code}")
        print(f"   URL может быть недоступен или файл не найден.")
        print(f"   Попробуйте скачать файл вручную с GitHub:")
        print(f"   https://github.com/dreuxx/Russian-Grammar-Error-Correction-Dataset")
        raise
    except Exception as e:
        print(f"\n❌ Ошибка при загрузке: {e}")
        raise


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Загрузка Russian Grammar Error Correction Dataset с GitHub"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Директория для сохранения датасета (по умолчанию: data)"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="russian_gec_dataset_final (1).csv",
        help="Имя файла датасета (по умолчанию: russian_gec_dataset_final (1).csv)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать файл, если он уже существует"
    )
    
    args = parser.parse_args()
    
    try:
        dataset_path = download_dataset(
            output_dir=args.output,
            filename=args.filename,
            force=args.force
        )
        print(f"\n✓ Датасет готов к использованию!")
        print(f"\nСледующий шаг: подготовка данных для обучения")
        print(f"  python prepare_data.py --input {dataset_path} --format csv")
    except Exception as e:
        print(f"\n❌ Не удалось загрузить датасет: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
