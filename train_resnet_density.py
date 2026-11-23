import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# ================= НАСТРОЙКИ =================
XML_PATH = "annotations.xml"
IMAGES_DIR = "potato_piles"
MODEL_SAVE_PATH = "potato_resnet18_best.pth"
IMG_SIZE = (512, 512)              # Оптимальный баланс детализации и нагрузки на Radeon 780M
EPOCHS = 120
BATCH_SIZE = 2
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
            samples.append((str(img_path), points))

    return samples


class PotatoResNetDataset(Dataset):
    def __init__(self, samples, img_size=(512, 512), augment=False):
        self.samples = samples
        self.img_size = img_size  # (W, H)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, points = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        pts = np.array(points, dtype=np.float32)

        # Мульти-масштабная аугментация (случайный кроп 60%–100% кадра)
        # Помогает модели одинаково точно считать и крупные, и мелкие клубни
        if self.augment and np.random.rand() > 0.3:
            crop_scale = np.random.uniform(0.6, 1.0)
            cw, ch = int(orig_w * crop_scale), int(orig_h * crop_scale)
            x0 = np.random.randint(0, max(1, orig_w - cw + 1))
            y0 = np.random.randint(0, max(1, orig_h - ch + 1))

            img = img[y0:y0 + ch, x0:x0 + cw]
            mask = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + cw) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + ch)
            pts = pts[mask] - np.array([x0, y0], dtype=np.float32)
            orig_h, orig_w = ch, cw

        target_w, target_h = self.img_size
        img_resized = cv2.resize(img, (target_w, target_h))

        # Масштабируем координаты на карту плотности (выход сети имеет шаг /8 -> 64x64 при входе 512x512)
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        out_w, out_h = target_w // 8, target_h // 8
        density = np.zeros((out_h, out_w), dtype=np.float32)

        for x, y in pts:
            nx = min(max(0, int((x * scale_x) / 8.0)), out_w - 1)
            ny = min(max(0, int((y * scale_y) / 8.0)), out_h - 1)
            density[ny, nx] += 1.0

        density = gaussian_filter(density, sigma=1.5) * DENSITY_SCALE

        if self.augment:
            if np.random.rand() > 0.5:
                img_resized = np.fliplr(img_resized).copy()
                density = np.fliplr(density).copy()
            if np.random.rand() > 0.5:
                img_resized = np.flipud(img_resized).copy()
                density = np.flipud(density).copy()

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        density_tensor = torch.from_numpy(density).unsqueeze(0)
        return img_tensor, density_tensor


class PotatoResNetCounter(nn.Module):
    """Энкодер на предобученном ResNet18 + легкий декодер плотности (шаг /8)."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Берем слои до layer2 включительно (уменьшение разрешения в 8 раз, 128 каналов)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,  # /4
            backbone.layer1,   # /4, 64 channels
            backbone.layer2    # /8, 128 channels
        )

        # Легкий декодер без тяжелых dilation (безопасно для AMD gfx1103)
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
        features = self.encoder(x)
        density = self.decoder(features)
        return density


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
        PotatoResNetDataset(train_samples, IMG_SIZE, augment=True),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        PotatoResNetDataset(val_samples, IMG_SIZE, augment=False),
        batch_size=1, shuffle=False
    )

    device = get_compute_device()
    model = PotatoResNetCounter().to(device)

    # Энкодер обучаем с меньшим шагом (LR * 0.2), чтобы не разрушить предобученные веса
    optimizer = torch.optim.Adam([
        {"params": model.encoder.parameters(), "lr": LR * 0.2},
        {"params": model.decoder.parameters(), "lr": LR}
    ])
    criterion = nn.MSELoss(reduction="mean")

    best_mae = float("inf")
    print("🚀 Начинаем обучение на предобученном ResNet18...")

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
            train_loss += loss.item()

        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                pred_count = preds.sum().item() / DENSITY_SCALE
                true_count = targets.sum().item() / DENSITY_SCALE
                val_mae += abs(pred_count - true_count)

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
