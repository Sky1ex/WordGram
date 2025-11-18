"""
Скрипт для проверки поддержки CUDA в PyTorch
"""
import torch
import sys

print("=" * 60)
print("Проверка поддержки CUDA в PyTorch")
print("=" * 60)

print(f"\nВерсия PyTorch: {torch.__version__}")
print(f"Версия Python: {sys.version}")

if "+cpu" in torch.__version__:
    print("\n⚠ ВНИМАНИЕ: Установлена версия PyTorch БЕЗ поддержки CUDA!")
    print("   (в версии есть '+cpu', что означает CPU-only версию)")
elif "+cu" in torch.__version__:
    print(f"\n✓ Установлена версия PyTorch С поддержкой CUDA")
    cuda_version = torch.__version__.split("+cu")[1].split("+")[0]
    print(f"   Версия CUDA в PyTorch: {cuda_version}")
else:
    print("\n⚠ Неясно, есть ли поддержка CUDA в PyTorch")

print(f"\nПроверка доступности CUDA:")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"\n✓ CUDA доступен!")
    print(f"  Версия CUDA: {torch.version.cuda}")
    print(f"  Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    Имя: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Память: {props.total_memory / 1e9:.2f} GB")
        print(f"    Вычислительная способность: {props.major}.{props.minor}")
else:
    print("\n❌ CUDA НЕ доступен!")
    print("\nВозможные причины:")
    print("  1. PyTorch установлен без поддержки CUDA (CPU-only версия)")
    print("  2. Драйверы NVIDIA не установлены")
    print("  3. CUDA toolkit не установлен или несовместим")
    print("  4. GPU не поддерживается")
    
    print("\nКак исправить:")
    print("\n1. Определите версию CUDA на вашей системе:")
    print("   - Откройте командную строку")
    print("   - Выполните: nvidia-smi")
    print("   - Посмотрите на версию CUDA в верхней строке")
    
    print("\n2. Переустановите PyTorch с поддержкой CUDA:")
    print("\n   Для CUDA 11.8:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n   Для CUDA 12.1:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n   Или используйте официальный сайт:")
    print("   https://pytorch.org/get-started/locally/")
    
    print("\n3. После переустановки запустите этот скрипт снова для проверки")

print("\n" + "=" * 60)



