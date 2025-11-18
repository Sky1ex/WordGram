#!/usr/bin/env python3
"""
Простой скрипт для создания placeholder иконок для расширения WordGram.
Требует библиотеку Pillow: pip install Pillow
"""

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Ошибка: требуется библиотека Pillow")
    print("Установите её командой: pip install Pillow")
    exit(1)

import os

def create_icon(size, output_path):
    """Создает простую иконку с текстом 'WG'"""
    # Создаем изображение с прозрачным фоном
    img = Image.new('RGBA', (size, size), (99, 102, 241, 255))  # Индиго цвет
    draw = ImageDraw.Draw(img)
    
    # Пытаемся использовать системный шрифт
    try:
        # Для Windows
        font_path = "C:/Windows/Fonts/arial.ttf"
        if not os.path.exists(font_path):
            # Для Linux
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            # Для macOS
            font_path = "/System/Library/Fonts/Helvetica.ttc"
        
        font_size = int(size * 0.6)
        font = ImageFont.truetype(font_path, font_size)
    except:
        # Используем стандартный шрифт, если не удалось загрузить
        font = ImageFont.load_default()
    
    # Рисуем текст 'WG' белым цветом
    text = "WG"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Центрируем текст
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - bbox[1]
    
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    
    # Сохраняем иконку
    img.save(output_path, 'PNG')
    print(f"Создана иконка: {output_path} ({size}x{size})")

def main():
    # Создаем папку icons, если её нет
    icons_dir = 'icons'
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
        print(f"Создана папка: {icons_dir}")
    
    # Создаем иконки разных размеров
    sizes = [16, 48, 128]
    for size in sizes:
        output_path = os.path.join(icons_dir, f'icon{size}.png')
        create_icon(size, output_path)
    
    print("\n✓ Все иконки успешно созданы!")
    print("Теперь вы можете загрузить расширение в браузер.")

if __name__ == '__main__':
    main()

