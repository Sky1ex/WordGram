"""
Конфигурационный файл для обучения модели исправления грамматики и пунктуации
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Конфигурация для обучения модели"""
    
    # Пути к данным
    train_data_path: str = "data/train.jsonl"
    val_data_path: str = "data/val.jsonl"
    test_data_path: Optional[str] = "data/test.jsonl"
    
    # Модель
    # ВАЖНО: Для 8GB GPU рекомендуется использовать ruT5-base вместо ruT5-large
    # model_name: str = "ai-forever/ruT5-base"  # Меньшая модель для 8GB GPU
    model_name: str = "ai-forever/ruT5-large"  # Большая модель (требует много памяти)
    output_dir: str = "models/grammar_corrector"
    
    # Параметры обучения
    batch_size: int = 1  # Уменьшено для 8GB GPU
    gradient_accumulation_steps: int = 16  # Увеличено для компенсации меньшего batch_size
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    max_seq_length: int = 256  # Уменьшено для экономии памяти
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_steps: int = 50  # Логирование каждые N шагов
    save_steps: Optional[int] = 100  # Сохранение каждые 100 шагов (или установите на None для сохранения каждую эпоху)
    eval_steps: Optional[int] = 100   # Оценка каждые 100 шагов (или установите на None для оценки каждую эпоху)
    save_total_limit: int = 10  # Максимальное количество сохраняемых чекпоинтов (старые автоматически удаляются)
    # Примечание: Если хотите сохранять каждую эпоху, установите save_steps=None
    
    # Параметры генерации (для валидации)
    max_length: int = 256  # Уменьшено для экономии памяти
    num_beams: int = 3  # Уменьшено для экономии памяти
    early_stopping: bool = True
    
    # Другие параметры
    seed: int = 42
    fp16: bool = True  # Использовать mixed precision для ускорения
    dataloader_num_workers: int = 0  # Отключено для экономии памяти GPU
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Промпт для обучения
    prefix: str = "Исправить грамматику и пунктуацию: "


# Глобальный экземпляр конфигурации
config = TrainingConfig()
