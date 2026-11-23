import os
import sys
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from train_resnet_native import PotatoResNetCounter, get_compute_device, pad_to_multiple, DENSITY_SCALE

MODEL_PATH = "potato_resnet18_native.pth"
IMAGE_PATH = "potatoes_2.jpg"
OUTPUT_PATH = "counted_result_native.jpg"

if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
    print(f"❌ Проверьте пути к файлам!\nМодель: {MODEL_PATH}\nФото: {IMAGE_PATH}")
    sys.exit(1)

device = get_compute_device()
model = PotatoResNetCounter().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def count_potatoes_native(img_path):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print("❌ Не удалось прочитать изображение.")
        return

    orig_h, orig_w = orig_img.shape[:2]
    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Подаем фото в родном разрешении 1:1 (только выравниваем края до кратности 8)
    padded_rgb = pad_to_multiple(rgb, multiple=8)

    img_tensor = torch.from_numpy(padded_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map = model(img_tensor)

    # 1. Интегральный подсчет по всему кадру
    predicted_count = density_map.sum().item() / DENSITY_SCALE
    print(f"\n🎉 РЕЗУЛЬТАТ ПОДСЧЕТА (Native ResNet18): ~{round(predicted_count)} картофелин (точное: {predicted_count:.2f})")

    # 2. Обрезаем поля дополнения (padding) на карте плотности обратно до размера оригинала
    d_map = density_map.squeeze().cpu().numpy()
    d_map_full = cv2.resize(d_map, (padded_rgb.shape[1], padded_rgb.shape[0]))
    d_map_cropped = d_map_full[:orig_h, :orig_w]

    # Накладываем тепловую карту
    d_map_norm = cv2.normalize(d_map_cropped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(d_map_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.72, heatmap, 0.28, 0)

    # Находим пики на уменьшенной карте плотности для визуализации точек
    peaks = (d_map == maximum_filter(d_map, size=3)) & (d_map > (d_map.max() * 0.15))
    ys, xs = np.where(peaks)
    for x, y in zip(xs, ys):
        cx, cy = int(x * 8 + 4), int(y * 8 + 4)
        if cx < orig_w and cy < orig_h:
            cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)

    cv2.putText(
        overlay, f"Native Count: {round(predicted_count)}", (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3
    )
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"💾 Результат сохранен в '{OUTPUT_PATH}'")


if __name__ == "__main__":
    count_potatoes_native(IMAGE_PATH)
    