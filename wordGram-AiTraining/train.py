"""
Скрипт для обучения модели исправления грамматики и пунктуации
"""
import os
import warnings
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
import numpy as np
from config import config
from data_utils import load_jsonl, create_hf_dataset

# Подавляем предупреждения TensorFlow и protobuf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*legacy.*')
warnings.filterwarnings('ignore', message='.*as_target_tokenizer.*')
warnings.filterwarnings('ignore', message='.*tokenizer.*deprecated.*')


class ProgressCallback(TrainerCallback):
    """Callback для отображения прогресса обучения"""
    
    def __init__(self):
        self.last_logged_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Вызывается при каждом логировании"""
        if logs is not None and state.global_step > self.last_logged_step:
            self.last_logged_step = state.global_step
            
            # Формируем строку с информацией
            info_parts = []
            
            if 'loss' in logs:
                info_parts.append(f"Loss: {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                info_parts.append(f"LR: {logs['learning_rate']:.2e}")
            if 'epoch' in logs:
                info_parts.append(f"Эпоха: {logs['epoch']:.2f}")
                
            if info_parts:
                print(f"\n[Шаг {state.global_step}/{state.max_steps}] " + " | ".join(info_parts))
        
    def on_epoch_end(self, args, state, control, **kwargs):
        """Вызывается в конце каждой эпохи"""
        print(f"\n{'='*60}")
        print(f"✓ Эпоха {int(state.epoch)}/{args.num_train_epochs} завершена!")
        if state.log_history:
            # Ищем последний лог с loss
            for log in reversed(state.log_history):
                if 'loss' in log:
                    print(f"  Текущий Loss: {log['loss']:.4f}")
                    break
        print(f"{'='*60}\n")
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Вызывается после оценки"""
        if state.log_history:
            # Ищем последний лог с eval метриками
            for log in reversed(state.log_history):
                if 'eval_loss' in log:
                    print(f"\n{'─'*60}")
                    print(f"[Оценка на шаге {state.global_step}]")
                    print(f"  Loss: {log['eval_loss']:.4f}")
                    if 'eval_exact_match' in log:
                        print(f"  Exact match: {log['eval_exact_match']:.4f}")
                    if 'eval_bleu' in log:
                        print(f"  BLEU: {log['eval_bleu']:.4f}")
                    print(f"{'─'*60}\n")
                    break

# Попытка импорта evaluate, если не доступен - используем fallback
try:
    import evaluate
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False


def compute_metrics_factory(tokenizer):
    """
    Создает функцию compute_metrics с доступом к токенизатору
    
    Args:
        tokenizer: Токенизатор модели
        
    Returns:
        Функция compute_metrics
    """
    def compute_metrics(eval_pred):
        """
        Вычисляет метрики для оценки модели
        
        Args:
            eval_pred: Предсказания модели
            
        Returns:
            Словарь с метриками
        """
        predictions, labels = eval_pred
        
        # Преобразуем в numpy array если это не так
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions, dtype=np.int64)
        else:
            predictions = predictions.astype(np.int64)
            
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels, dtype=np.int64)
        else:
            labels = labels.astype(np.int64)
        
        # Если predictions имеют 3 измерения (batch, seq_len, vocab_size), 
        # берем argmax по последней оси
        if len(predictions.shape) > 2:
            predictions = np.argmax(predictions, axis=-1)
        
        # Получаем размер словаря для ограничения значений
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # Если pad_token_id равен None, используем eos_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        
        # Обрабатываем predictions: заменяем все недопустимые значения
        # 1. Заменяем отрицательные значения на pad_token_id
        predictions = np.where(predictions < 0, pad_token_id, predictions)
        # 2. Заменяем значения >= vocab_size на pad_token_id
        predictions = np.where(predictions >= vocab_size, pad_token_id, predictions)
        # 3. Преобразуем в int32 для совместимости с токенизатором
        predictions = predictions.astype(np.int32)
        
        # Обрабатываем labels: заменяем все недопустимые значения
        # 1. Заменяем -100 (игнорируемые токены) на pad_token_id
        labels = np.where(labels == -100, pad_token_id, labels)
        # 2. Заменяем отрицательные значения на pad_token_id
        labels = np.where(labels < 0, pad_token_id, labels)
        # 3. Заменяем значения >= vocab_size на pad_token_id
        labels = np.where(labels >= vocab_size, pad_token_id, labels)
        # 4. Преобразуем в int32 для совместимости с токенизатором
        labels = labels.astype(np.int32)
        
        # Декодируем predictions
        # Преобразуем numpy array в список списков (Python list) для токенизатора
        # Это необходимо, так как некоторые токенизаторы не принимают numpy arrays напрямую
        if isinstance(predictions, np.ndarray):
            predictions_list = predictions.tolist()
        else:
            predictions_list = predictions
        
        try:
            decoded_preds = tokenizer.batch_decode(predictions_list, skip_special_tokens=True)
        except Exception as e:
            # Если все еще есть ошибка, попробуем более безопасный способ
            print(f"⚠ Предупреждение при декодировании predictions: {e}")
            # Дополнительная очистка: заменяем все проблемные значения
            predictions_clean = np.where((predictions < 0) | (predictions >= vocab_size), pad_token_id, predictions)
            decoded_preds = tokenizer.batch_decode(predictions_clean.astype(np.int32).tolist(), skip_special_tokens=True)
        
        # Декодируем labels
        if isinstance(labels, np.ndarray):
            labels_list = labels.tolist()
        else:
            labels_list = labels
            
        try:
            decoded_labels = tokenizer.batch_decode(labels_list, skip_special_tokens=True)
        except Exception as e:
            print(f"⚠ Предупреждение при декодировании labels: {e}")
            # Дополнительная очистка
            labels_clean = np.where((labels < 0) | (labels >= vocab_size), pad_token_id, labels)
            decoded_labels = tokenizer.batch_decode(labels_clean.astype(np.int32).tolist(), skip_special_tokens=True)
        
        # Вычисляем метрики (можно добавить BLEU, ROUGE и т.д.)
        # Пока используем простую точность совпадения
        exact_match = sum(p == l for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)
        
        # Вычисляем BLEU score (если доступен)
        bleu_score = 0.0
        if EVALUATE_AVAILABLE:
            try:
                bleu_metric = evaluate.load("sacrebleu")
                bleu_results = bleu_metric.compute(
                    predictions=decoded_preds,
                    references=[[ref] for ref in decoded_labels]
                )
                bleu_score = bleu_results["score"] / 100.0
            except Exception:
                # Если не удалось загрузить метрику, просто пропускаем
                pass
        
        return {
            "exact_match": exact_match,
            "bleu": bleu_score
        }
    
    return compute_metrics


def main():
    """Основная функция обучения"""
    
    print("=" * 60)
    print("Обучение модели исправления грамматики и пунктуации")
    print("=" * 60)
    
    # Очищаем память GPU перед началом (на случай предыдущих запусков)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ Память GPU очищена")
    
    # Проверяем наличие GPU и CUDA
    print("\nПроверка CUDA...")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA доступен в PyTorch: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        try:
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Количество GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Память: {props.total_memory / 1e9:.2f} GB")
                # Показываем свободную память
                if torch.cuda.is_available():
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1e9
                    print(f"    Выделено: {allocated:.2f} GB")
                    print(f"    Зарезервировано: {reserved:.2f} GB")
                    print(f"    Свободно: {free:.2f} GB")
            device = "cuda"
            print(f"\n✓ Используется GPU: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"  ⚠ Ошибка при получении информации о GPU: {e}")
            print("  Переключение на CPU...")
            device = "cpu"
    else:
        print("  ⚠ CUDA недоступен!")
        print("  Возможные причины:")
        print("    1. PyTorch установлен без поддержки CUDA")
        print("    2. Драйверы NVIDIA не установлены или устарели")
        print("    3. CUDA toolkit не установлен или несовместим")
        print("\n  Рекомендации:")
        print("    - Установите PyTorch с CUDA: https://pytorch.org/get-started/locally/")
        print("    - Для CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("    - Для CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n  Переключение на CPU...")
        device = "cpu"
    
    if device == "cpu":
        print("\n⚠ ВНИМАНИЕ: Обучение на CPU будет очень медленным!")
        print("   Рекомендуется использовать GPU для ускорения обучения.")
        print("   Ожидаемое время обучения на CPU: несколько часов или дней.")
        print("   Рассмотрите возможность использования ruT5-base вместо ruT5-large.\n")
    
    # Загружаем токенизатор и модель
    print(f"\n[1/5] Загрузка модели {config.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Перемещаем модель на нужное устройство
        if device == "cuda" and torch.cuda.is_available():
            try:
                model = model.to(device)
                print(f"✓ Модель перемещена на GPU")
            except RuntimeError as e:
                print(f"⚠ Не удалось переместить модель на GPU: {e}")
                print("  Переключение на CPU...")
                device = "cpu"
                model = model.to("cpu")
        else:
            model = model.to("cpu")
        
        # Устанавливаем pad_token, если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # Включаем gradient checkpointing для экономии памяти (торгуем память на скорость)
        if device == "cuda" and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing включен (экономия памяти)")
            except Exception as e:
                print(f"⚠ Не удалось включить gradient checkpointing: {e}")
        
        print("✓ Модель загружена и готова к обучению")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Проверяем наличие evaluate для метрик
    if not EVALUATE_AVAILABLE:
        print("⚠ Предупреждение: библиотека 'evaluate' не установлена. BLEU метрика будет недоступна.")
        print("   Рекомендуется установить: pip install evaluate")
        print("   Обучение продолжится, но BLEU score не будет вычисляться.\n")
    
    # Создаем функцию compute_metrics с доступом к токенизатору
    compute_metrics = compute_metrics_factory(tokenizer)
    
    # Загружаем данные
    print(f"\n[2/5] Загрузка данных...")
    try:
        if not os.path.exists(config.train_data_path):
            raise FileNotFoundError(f"Файл с обучающими данными не найден: {config.train_data_path}")
        if not os.path.exists(config.val_data_path):
            raise FileNotFoundError(f"Файл с валидационными данными не найден: {config.val_data_path}")
            
        train_data = load_jsonl(config.train_data_path)
        val_data = load_jsonl(config.val_data_path)
        test_data = load_jsonl(config.test_data_path) if config.test_data_path and os.path.exists(config.test_data_path) else None
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        print(f"   Убедитесь, что файлы данных существуют и правильно подготовлены.")
        print(f"   Запустите: python prepare_data.py")
        raise
    
    print(f"  Обучающих примеров: {len(train_data)}")
    print(f"  Валидационных примеров: {len(val_data)}")
    if test_data:
        print(f"  Тестовых примеров: {len(test_data)}")
    
    if len(train_data) == 0:
        raise ValueError("Обучающий датасет пуст! Проверьте путь к данным.")
    
    # Создаем HuggingFace датасеты
    print(f"\n[3/5] Подготовка датасетов...")
    try:
        print("  Токенизация данных (это может занять некоторое время)...")
        datasets = create_hf_dataset(
            train_data=train_data,
            val_data=val_data,
            tokenizer=tokenizer,
            config=config,
            test_data=test_data
        )
        print("✓ Датасеты подготовлены")
        
        # Проверяем размер датасетов
        print(f"  Размер обучающего датасета: {len(datasets['train'])}")
        print(f"  Размер валидационного датасета: {len(datasets['validation'])}")
        if 'test' in datasets:
            print(f"  Размер тестового датасета: {len(datasets['test'])}")
    except Exception as e:
        print(f"❌ Ошибка при подготовке датасетов: {e}")
        import traceback
        traceback.print_exc()
        print("\nВозможные причины:")
        print("  1. Нехватка памяти (RAM)")
        print("  2. Поврежденные данные")
        print("  3. Проблемы с токенизацией")
        print("\nПопробуйте:")
        print("  - Уменьшить размер датасета")
        print("  - Уменьшить max_seq_length в config.py")
        print("  - Закрыть другие приложения для освобождения памяти")
        raise
    
    # Настройки обучения
    print(f"\n[4/5] Настройка параметров обучения...")
    
    # Для CPU уменьшаем частоту логирования и количество воркеров
    if device == "cpu":
        logging_steps = max(10, config.logging_steps // 10)  # Логируем чаще на CPU
        dataloader_workers = 0  # На CPU лучше не использовать воркеры
        dataloader_pin_memory = False  # Отключаем pin_memory на CPU
        print(f"   Настройки для CPU: logging_steps={logging_steps}, workers={dataloader_workers}")
    else:
        logging_steps = config.logging_steps
        dataloader_workers = config.dataloader_num_workers
        dataloader_pin_memory = True
    
    # Вычисляем общее количество шагов для информации
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = (len(train_data) + effective_batch_size - 1) // effective_batch_size
    total_steps = steps_per_epoch * config.num_train_epochs
    print(f"   Всего шагов обучения: ~{total_steps}")
    print(f"   Шагов на эпоху: ~{steps_per_epoch}")
    
    # Вычисляем частоту сохранений
    if config.save_steps:
        saves_per_epoch = steps_per_epoch // config.save_steps
        total_saves = (total_steps // config.save_steps) + 1
        print(f"   Сохранений на эпоху: ~{saves_per_epoch}")
        print(f"   Всего сохранений: ~{total_saves}")
    else:
        print(f"   Сохранение: каждую эпоху ({config.num_train_epochs} сохранений)")
    
    # Проверяем размер модели и доступную память
    if device == "cuda" and torch.cuda.is_available():
        try:
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            free_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   Размер модели: ~{model_size_mb:.0f} MB")
            print(f"   Свободная память GPU: ~{free_memory_gb:.2f} GB")
            
            # Предупреждение о возможной нехватке памяти
            estimated_memory_gb = (model_size_mb / 1024) * 4  # Примерная оценка
            if estimated_memory_gb > free_memory_gb * 0.8:
                print(f"   ⚠ Предупреждение: Модель может не поместиться в память GPU")
                print(f"      Рекомендуется уменьшить batch_size или использовать CPU")
        except:
            pass
    
    # Определяем стратегию сохранения и оценки
    # Если save_steps не указан (None), сохраняем каждую эпоху
    if config.save_steps is None:
        save_strategy = "epoch"
        save_steps_value = None
        eval_strategy = "epoch"
        eval_steps_value = None
        print(f"   Стратегия сохранения: каждую эпоху")
    else:
        save_strategy = "steps"
        save_steps_value = config.save_steps
        eval_strategy = "steps"
        eval_steps_value = config.eval_steps
        print(f"   Стратегия сохранения: каждые {config.save_steps} шагов")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=logging_steps,
        save_steps=save_steps_value,
        eval_steps=eval_steps_value,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,  # для loss меньше = лучше
        fp16=config.fp16 and device == "cuda",
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        predict_with_generate=True,
        generation_max_length=config.max_length,
        generation_num_beams=config.num_beams,
        seed=config.seed,
        report_to="none",  # Можно изменить на "tensorboard" или "wandb"
        logging_first_step=True,  # Логируем первый шаг
        include_inputs_for_metrics=False,  # Ускоряет обучение
        gradient_checkpointing=True,  # Экономия памяти за счет скорости
        dataloader_drop_last=True,  # Отбрасываем последний неполный батч
    )
    
    # Data collator для seq2seq задач
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    print("✓ Параметры обучения настроены")
    
    # Создаем trainer
    print(f"\n[5/5] Инициализация trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if "validation" in datasets else None,
        callbacks=[ProgressCallback()],  # Добавляем callback для прогресса
    )
    
    print("✓ Trainer инициализирован")
    
    # Начинаем обучение
    print("\n" + "=" * 60)
    print("Начало обучения...")
    print("=" * 60)
    print(f"\nПараметры обучения:")
    print(f"  - Модель: {config.model_name}")
    print(f"  - Устройство: {device}")
    print(f"  - Эпох: {config.num_train_epochs}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  - Эффективный batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  - Всего примеров: {len(train_data)}")
    print(f"  - Примеров на эпоху: {len(train_data)}")
    if config.save_steps:
        print(f"  - Сохранение: каждые {config.save_steps} шагов")
        print(f"  - Оценка: каждые {config.eval_steps} шагов")
    else:
        print(f"  - Сохранение: каждую эпоху")
        print(f"  - Оценка: каждую эпоху")
    print(f"  - Максимум чекпоинтов: {config.save_total_limit}")
    print(f"\nОбучение начато. Ожидайте обновлений прогресса...")
    if device == "cpu":
        print(f"⚠ ПРИМЕЧАНИЕ: Первый шаг может занять несколько минут из-за компиляции.")
        print(f"   После этого процесс ускорится. Пожалуйста, подождите...\n")
    print("=" * 60 + "\n")
    
    try:
        train_result = trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            print(f"\n\n❌ Ошибка GPU памяти: {e}")
            
            # Очищаем память GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ Память GPU очищена")
            
            print("\n" + "="*60)
            print("РЕКОМЕНДАЦИИ ДЛЯ РЕШЕНИЯ ПРОБЛЕМЫ С ПАМЯТЬЮ:")
            print("="*60)
            print("\n1. УМЕНЬШИТЕ BATCH_SIZE:")
            print(f"   Текущее значение: {config.batch_size}")
            print("   Рекомендуемое: batch_size = 1")
            print("   И увеличьте gradient_accumulation_steps до 16-32")
            print("\n2. ИСПОЛЬЗУЙТЕ МЕНЬШУЮ МОДЕЛЬ:")
            print("   В config.py измените:")
            print("   model_name = 'ai-forever/ruT5-base'  # вместо ruT5-large")
            print("\n3. УМЕНЬШИТЕ ДЛИНУ ПОСЛЕДОВАТЕЛЬНОСТЕЙ:")
            print(f"   Текущее значение: max_seq_length = {config.max_seq_length}")
            print("   Рекомендуемое: max_seq_length = 256 или даже 128")
            print("\n4. ОЧИСТИТЕ ПАМЯТЬ GPU ПЕРЕД ЗАПУСКОМ:")
            print("   - Закройте другие программы, использующие GPU")
            print("   - Перезапустите Python скрипт")
            print("   - Или выполните: torch.cuda.empty_cache()")
            print("\n5. ИСПОЛЬЗУЙТЕ GRADIENT CHECKPOINTING:")
            print("   Уже включено в training_args (gradient_checkpointing=True)")
            print("\n6. РАССМОТРИТЕ ОБУЧЕНИЕ НА CPU:")
            print("   Если GPU недостаточно, обучение на CPU все равно возможно,")
            print("   но будет значительно медленнее")
            print("="*60)
            
            print("\nПопытка сохранить модель...")
            try:
                # Очищаем память перед сохранением
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trainer.save_model()
                tokenizer.save_pretrained(config.output_dir)
                print(f"✓ Модель сохранена в: {config.output_dir}")
            except Exception as save_error:
                print(f"⚠ Не удалось сохранить модель: {save_error}")
        else:
            print(f"\n\n❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
        raise
    except KeyboardInterrupt:
        print("\n\n⚠ Обучение прервано пользователем.")
        print("Модель будет сохранена в текущем состоянии...")
        try:
            trainer.save_model()
            tokenizer.save_pretrained(config.output_dir)
            print(f"Модель сохранена в: {config.output_dir}")
        except Exception as e:
            print(f"⚠ Не удалось сохранить модель: {e}")
        return
    except Exception as e:
        print(f"\n\n❌ Неожиданная ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Сохраняем финальную модель
    print("\nСохранение модели...")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    # Сохраняем метрики
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print("=" * 60)
    print(f"\nМодель сохранена в: {config.output_dir}")
    print(f"Финальная loss: {metrics.get('train_loss', 'N/A'):.4f}")
    
    # Оценка на валидационном наборе
    if "validation" in datasets:
        print("\nОценка на валидационном наборе...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        print(f"Валидационная loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        print(f"Exact match: {eval_metrics.get('eval_exact_match', 'N/A'):.4f}")
        print(f"BLEU score: {eval_metrics.get('eval_bleu', 'N/A'):.4f}")
    
    # Оценка на тестовом наборе, если есть
    if "test" in datasets:
        print("\nОценка на тестовом наборе...")
        test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        print(f"Тестовая loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
        print(f"Exact match: {test_metrics.get('test_exact_match', 'N/A'):.4f}")
        print(f"BLEU score: {test_metrics.get('test_bleu', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
