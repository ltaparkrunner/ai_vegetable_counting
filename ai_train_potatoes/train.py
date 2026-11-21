import os
from ultralytics import YOLO

def start_training():
    model = YOLO("yolov8n-seg.pt") 
    
    yaml_paths = ("ai_train_potatoes/data.yaml", "data.yaml")
    yaml_path = next((p for p in yaml_paths if os.path.exists(p)), None)
    
    if not yaml_path:
        print(f"❌ Ошибка: Ни один из файлов {yaml_paths} не найден!")
        return

    print(f"🚀 Начинаем обучение на {yaml_path}...")
    
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        amp=False,        # Ключевой параметр: отключает проблемный FP16 на Radeon 780M
        workers=2,
        mosaic=1.0
    )
    
    print("\n🎉 Обучение успешно завершено!")

if __name__ == "__main__":
    start_training()
