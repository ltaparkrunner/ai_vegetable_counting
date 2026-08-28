import cv2
from ultralytics import YOLO
import numpy as np
import os
import sys

# Load a lightweight pre-trained YOLO model (or a custom-trained potato model)
model_path = "ai_train_potatoes/runs/segment/train/weights/best.pt"
image_path = "potatoes_2.jpg"
if not os.path.exists(model_path) or not os.path.exists(image_path):
    print("❌ Ошибка: Проверьте правильность путей к модели или изображению!")
    sys.exit()  # The program stops here


def count_potatoes(image_path):
    # 1. Загрузка модели сегментации и предсказание
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25, iou=0.40)
    res = results[0] # Извлекаем результат для первого изображения

    # 2. Загружаем ОРИГИНАЛЬНОЕ ЧИСТОЕ фото для рисования контуров
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not read image.")
        return
    
    # 3. Проверяем, нашла ли модель полигональные маски
    if res.masks is not None:
        # Считаем точное количество найденных масок
        potato_count = len(res.masks)
        print(f"\n🎉 РЕЗУЛЬТАТ ПОДСЧЕТА: Найдено {potato_count} картофелин(ы)!")

        # Извлекаем сырые координаты полигонов (в пикселях)
        # res.masks.xy — это список массивов, где каждый массив — это точки (x, y) одной картофелины
        polygons = res.masks.xy

        # 4. Перебираем каждый полигон и рисуем только его границу
        for poly in polygons:
            if len(poly) == 0:
                continue
            
            # Переводим координаты точек в формат целых чисел для OpenCV
            points = np.array(poly, dtype=np.int32)
            
            # ВАРИАНТ А: Каждый контур случайного яркого цвета (чтобы они не сливались)
            color = (
                int(np.random.randint(50, 255)), 
                int(np.random.randint(50, 255)), 
                int(np.random.randint(50, 255))
            )
            
            # ВАРИАНТ Б: Если хотите, чтобы все контуры были строго одного цвета (например, зеленого),
            # раскомментируйте строчку ниже, а строчку выше со случайным цветом удалите:
            # color = (0, 255, 0) # Зеленый цвет в формате BGR

            # Рисуем замкнутую полигональную линию (границу)
            # thickness=2 — это толщина линии в пикселях. Можно сделать толще (3, 4) или тоньше (1).
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
            
    else:
        print("\n❌ Нейросеть не обнаружила объектов. Проверьте параметры conf/iou.")

    # 5. Сохраняем результат
    output_path = "counted_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"💾 Результат с тонкими полигональными границами сохранен в '{output_path}'")

# Run the function
count_potatoes(image_path)
