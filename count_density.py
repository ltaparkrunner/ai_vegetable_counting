import os
import sys
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from train_dens_2 import PotatoCSRNet, get_compute_device, IMG_SIZE, DENSITY_SCALE

MODEL_PATH = "potato_density_best.pth"
IMAGE_PATH = "potatoes_2.jpg"
OUTPUT_PATH = "counted_result_density.jpg"

if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
    print(f"❌ Проверьте пути к файлам!\nМодель: {MODEL_PATH}\nФото: {IMAGE_PATH}")
    sys.exit(1)

device = get_compute_device()
model = PotatoCSRNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


def count_potatoes_density(img_path):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print("❌ Не удалось прочитать изображение.")
        return

    orig_h, orig_w = orig_img.shape[:2]
    rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)

    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = ((img_tensor - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map = model(img_tensor)

    # 1. Интегральный подсчет (сумма плотности / DENSITY_SCALE)
    predicted_count = density_map.sum().item() / DENSITY_SCALE
    print(f"\n🎉 РЕЗУЛЬТАТ ПОДСЧЕТА (Density Map): ~{round(predicted_count)} картофелин (точное: {predicted_count:.2f})")

    # 2. Визуализация: накладываем полупрозрачную тепловую карту + ставим точки в пиках
    d_map = density_map.squeeze().cpu().numpy()
    d_map_norm = cv2.normalize(d_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d_map_resized = cv2.resize(d_map_norm, (orig_w, orig_h))
    heatmap = cv2.applyColorMap(d_map_resized, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(orig_img, 0.70, heatmap, 0.30, 0)

    # Находим локальные максимумы на карте плотности, чтобы отметить центры клубней точками
    peaks = (d_map == maximum_filter(d_map, size=3)) & (d_map > (d_map.max() * 0.15))
    ys, xs = np.where(peaks)
    scale_x = orig_w / d_map.shape
    scale_y = orig_h / d_map.shape

    for x, y in zip(xs, ys):
        cx, cy = int((x + 0.5) * scale_x), int((y + 0.5) * scale_y)
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)

    cv2.putText(
        overlay, f"Estimated count: {round(predicted_count)}", (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3
    )
    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"💾 Результат сохранен в '{OUTPUT_PATH}'")


if __name__ == "__main__":
    count_potatoes_density(IMAGE_PATH)
