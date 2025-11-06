# Скрипт установки зависимостей для WordGram AI Backend
# Для Windows PowerShell

Write-Host "WordGram AI Backend - Установка зависимостей" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Проверка версии Python
$pythonVersion = python --version 2>&1
Write-Host "Обнаружен Python: $pythonVersion" -ForegroundColor Yellow
Write-Host ""

# Обновление pip
Write-Host "Обновление pip..." -ForegroundColor Green
python -m pip install --upgrade pip
Write-Host ""

# Попытка установки с предпочтением бинарных пакетов
Write-Host "Установка зависимостей (предпочтение бинарным пакетам)..." -ForegroundColor Green
Write-Host ""

# Установка по отдельности для лучшей обработки ошибок
$packages = @(
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "transformers>=4.35.0",
    "pydantic>=2.6.0",
    "python-multipart>=0.0.6",
    "tiktoken>=0.5.0",
    "sentencepiece>=0.1.99"
)

foreach ($package in $packages) {
    Write-Host "Установка $package..." -ForegroundColor Yellow
    python -m pip install --prefer-binary $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Ошибка при установке $package" -ForegroundColor Red
        Write-Host "Попытка установки без предпочтения бинарных пакетов..." -ForegroundColor Yellow
        python -m pip install $package
    }
}

Write-Host ""
Write-Host "Установка PyTorch (CPU версия)..." -ForegroundColor Yellow
# Установка PyTorch с официального сайта (CPU версия для Windows)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Установка завершена!" -ForegroundColor Green
Write-Host ""
Write-Host "Если были ошибки с pydantic-core:" -ForegroundColor Yellow
Write-Host "1. Установите Rust: https://rustup.rs/" -ForegroundColor White
Write-Host "2. Или используйте Python 3.11/3.12" -ForegroundColor White
Write-Host ""
