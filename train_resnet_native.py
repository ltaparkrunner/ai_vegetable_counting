import os
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

import time

# ================= НАСТРОЙКИ =================
XML_PATH = "annotations.xml"
IMAGES_DIR = "potato_piles"
MODEL_SAVE_PATH = "potato_resnet18_native.pth"
CROP_SIZE = 256                    # Размер вырезаемого квадрата в родном разрешении (1:1)
CROPS_PER_IMAGE = 4 #6                # Сколько случайных фрагментов брать с каждого фото за эпоху
EPOCHS = 60 #100
BATCH_SIZE = 2 #4
LR = 3e-4
DENSITY_SCALE = 100.0
# =============================================


def get_compute_device():
    """Проверяет наличие GPU и работоспособность сверток (защита для AMD gfx1103)."""
    if not torch.cuda.is_available():
        print("⚙️ GPU не найден. Используем CPU.")
        return torch.device("cpu")
    try:
        name = torch.cuda.get_device_name(0)
        x = torch.randn(1, 3, 32, 32, device="cuda")
        conv = nn.Conv2d(3, 8, 3).to("cuda")
        _ = conv(x)
        torch.cuda.synchronize()
        print(f"⚙️ GPU проверен и работает: {name}")
        return torch.device("cuda:0")
    except Exception as e:
        print(f"⚠️ Ошибка GPU ({e}). Переключаемся на CPU.")
        return torch.device("cpu")


def parse_cvat_xml(xml_path, images_dir):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples = []

    for image_tag in root.findall("image"):
        file_name = Path(image_tag.get("name")).name
        img_path = Path(images_dir) / file_name
        if not img_path.exists():
            continue

        points = []
        for pts_tag in image_tag.findall("points"):
            raw_pts = pts_tag.get("points").strip().split(";")
            for pair in raw_pts:
                if "," in pair:
                    x_str, y_str = pair.split(",")
                    points.append((float(x_str), float(y_str)))

        if len(points) > 0:
            samples.append((str(img_path), np.array(points, dtype=np.float32)))

    return samples


def build_adaptive_density_map(img_h, img_w, points, stride=8):
    """
    Строит карту плотности в родном разрешении без сжатия.
    Размер пятна sigma подбирается автоматически по расстоянию до 3 ближайших соседей.
    """
    out_h = img_h // stride
    out_w = img_w // stride
    density = np.zeros((out_h, out_w), dtype=np.float32)

    if len(points) == 0:
        return density

    # Вычисляем среднее расстояние до 3 ближайших соседей для адаптивной sigma
    if len(points) >= 4:
        tree = KDTree(points)
        dists, _ = tree.query(points, k=4)  # k=4, так как 1-й сосед — сама точка (dist=0)
        avg_nn_dists = np.mean(dists[:, 1:], axis=1)
    else:
        avg_nn_dists = np.full(len(points), 24.0, dtype=np.float32)

    for (x, y), nn_dist in zip(points, avg_nn_dists):
        nx = min(max(0, int(x / stride)), out_w - 1)
        ny = min(max(0, int(y / stride)), out_h - 1)

        point_map = np.zeros((out_h, out_w), dtype=np.float32)
        point_map[ny, nx] = 1.0

        # Адаптивная sigma в координатах карты плотности (ограничена разумными пределами)
        sigma = np.clip((nn_dist * 0.15) / stride, 0.8, 3.0)
        density += gaussian_filter(point_map, sigma=sigma)

    return density * DENSITY_SCALE


def pad_to_multiple(img, multiple=8):
    """Дополняет фото справа и снизу нулями, чтобы размеры делились на 8 без искажения масштаба."""
    h, w = img.shape[:2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img


class NativeCropPotatoDataset(Dataset):
    def __init__(self, samples, crop_size=256, crops_per_image=6, augment=True):
        self.samples = samples
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image if augment else 1
        self.augment = augment

    def __len__(self):
        return len(self.samples) * self.crops_per_image

    def __getitem__(self, idx):
        sample_idx = idx // self.crops_per_image
        img_path, points = self.samples[sample_idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pts = points.copy()

        # Небольшая вариация масштаба (0.85x - 1.15x) для устойчивости к высоте камеры
        if self.augment and np.random.rand() > 0.3:
            scale = np.random.uniform(0.85, 1.15)
            new_w = max(self.crop_size, int(img.shape[1] * scale))
            new_h = max(self.crop_size, int(img.shape[0] * scale))
            img = cv2.resize(img, (new_w, new_h))
            pts = pts * scale

        h, w = img.shape[:2]

        if self.augment:
            # Дополняем если картинка меньше CROP_SIZE
            if h < self.crop_size or w < self.crop_size:
                pad_h = max(0, self.crop_size - h)
                pad_w = max(0, self.crop_size - w)
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                h, w = img.shape[:2]

            # Вырезаем случайный квадрат CROP_SIZE x CROP_SIZE в родном разрешении
            x0 = np.random.randint(0, w - self.crop_size + 1)
            y0 = np.random.randint(0, h - self.crop_size + 1)
            crop_img = img[y0:y0 + self.crop_size, x0:x0 + self.crop_size]

            mask = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + self.crop_size) & \
                   (pts[:, 1] >= y0) & (pts[:, 1] < y0 + self.crop_size)
            crop_pts = pts[mask] - np.array([x0, y0], dtype=np.float32)

            # Случайные повороты на 90 градусов и отражения
            k = np.random.randint(0, 4)
            if k > 0:
                crop_img = np.rot90(crop_img, k).copy()
                for _ in range(k):
                    # Поворот координат (x, y) против часовой стрелки на 90°
                    new_x = crop_pts[:, 1].copy()
                    new_y = self.crop_size - 1 - crop_pts[:, 0].copy()
                    crop_pts = np.stack([new_x, new_y], axis=1) if len(crop_pts) > 0 else crop_pts

            if np.random.rand() > 0.5:
                crop_img = np.fliplr(crop_img).copy()
                if len(crop_pts) > 0:
                    crop_pts[:, 0] = self.crop_size - 1 - crop_pts[:, 0]

            density = build_adaptive_density_map(self.crop_size, self.crop_size, crop_pts, stride=8)
            final_img = crop_img
        else:
            # Для валидации используем полное изображение без сжатия (дополненное до кратности 8)
            final_img = pad_to_multiple(img, multiple=8)
            density = build_adaptive_density_map(final_img.shape[0], final_img.shape[1], pts, stride=8)

        img_tensor = torch.from_numpy(final_img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        density_tensor = torch.from_numpy(density).unsqueeze(0)
        return img_tensor, density_tensor


class PotatoResNetCounter(nn.Module):
    """Полносверточная сеть на ResNet18 — принимает фото любого разрешения без сжатия."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # /4
            backbone.layer1,   # /4, 64 channels
            backbone.layer2    # /8, 128 channels
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train():
    if not os.path.exists(XML_PATH):
        print(f"❌ Файл '{XML_PATH}' не найден!")
        return

    samples = parse_cvat_xml(XML_PATH, IMAGES_DIR)
    print(f"📂 Найдено размеченных фото: {len(samples)}")
    if len(samples) == 0:
        return

    np.random.seed(42)
    np.random.shuffle(samples)
    split = max(1, int(len(samples) * 0.85))
    train_samples = samples[:split]
    val_samples = samples[split:] if len(samples) > 1 else samples

    train_loader = DataLoader(
        NativeCropPotatoDataset(train_samples, CROP_SIZE, CROPS_PER_IMAGE, augment=True),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        NativeCropPotatoDataset(val_samples, CROP_SIZE, crops_per_image=1, augment=False),
        batch_size=1, shuffle=False
    )

    device = get_compute_device()
    model = PotatoResNetCounter().to(device)

    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR * 0.2},
        {"params": model.decoder.parameters(), "lr": LR}
    ])
    criterion = nn.MSELoss(reduction="mean")

    best_mae = float("inf")
    print("🚀 Начинаем обучение в родном разрешении (без сжатия фото)...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            time.sleep(0.02)
            train_loss += loss.item()

        torch.cuda.empty_cache()
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                pred_count = preds.sum().item() / DENSITY_SCALE
                true_count = targets.sum().item() / DENSITY_SCALE
                val_mae += abs(pred_count - true_count)

        # Compute average MAE across all validation images AFTER the loop
        val_mae /= len(val_loader)
        if val_mae <= best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Эпоха {epoch:03d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Ошибка (MAE): ±{val_mae:.1f} шт.")

    print(f"\n🎉 Обучение завершено! Лучшая средняя ошибка: ±{best_mae:.1f} картофелин.")
    print(f"💾 Модель сохранена в '{MODEL_SAVE_PATH}'")


if __name__ == "__main__":
    train()