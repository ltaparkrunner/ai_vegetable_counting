import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= НАСТРОЙКИ =================
XML_PATH = "annotations.xml"       # Путь к скачанному файлу из CVAT
IMAGES_DIR = "potato_piles"       # Папка с вашими фотографиями
MODEL_SAVE_PATH = "potato_density_best.pth"
IMG_SIZE = (512, 512)              # Размер приведения фото (ширина, высота)
EPOCHS = 80                        # Количество эпох обучения
BATCH_SIZE = 4
LR = 1e-4
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
    """Читает annotations.xml из CVAT и собирает список (путь_к_фото, список_точек)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    samples = []

    for image_tag in root.findall("image"):
        file_name = Path(image_tag.get("name")).name
        img_path = Path(images_dir) / file_name
        if not img_path.exists():
            continue

        points = []
        # Поддерживает как отдельные <points>, так и несколько точек внутри одного тега <points>
        for pts_tag in image_tag.findall("points"):
            raw_pts = pts_tag.get("points").strip().split(";")
            for pair in raw_pts:
                if "," in pair:
                    x_str, y_str = pair.split(",")
                    points.append((float(x_str), float(y_str)))

        if len(points) > 0:
            samples.append((str(img_path), points))

    return samples


class PotatoDensityDataset(Dataset):
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

        target_w, target_h = self.img_size
        img_resized = cv2.resize(img, (target_w, target_h))

        # Масштабируем координаты точек под новый размер картинки
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        # Карта плотности уменьшена в 8 раз (выход сети 64x64 для входа 512x512)
        out_w, out_h = target_w // 8, target_h // 8
        density = np.zeros((out_h, out_w), dtype=np.float32)

        for x, y in points:
            nx = min(int((x * scale_x) / 8.0), out_w - 1)
            ny = min(int((y * scale_y) / 8.0), out_h - 1)
            density[ny, nx] += 1.0

        # Размываем каждую точку фильтром Гаусса (сумма пикселей остается равна числу картошек!)
        density = gaussian_filter(density, sigma=1.5)

        # Простая аугментация: случайное отражение по горизонтали и вертикали
        if self.augment:
            if np.random.rand() > 0.5:
                img_resized = np.fliplr(img_resized).copy()
                density = np.fliplr(density).copy()
            if np.random.rand() > 0.5:
                img_resized = np.flipud(img_resized).copy()
                density = np.flipud(density).copy()

        # Нормализация изображения
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        density_tensor = torch.from_numpy(density).unsqueeze(0)
        return img_tensor, density_tensor


class PotatoCSRNet(nn.Module):
    """Компактная архитектура CSRNet для подсчета плотных скоплений."""
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        # Расширенные свертки (Dilated Conv) для захвата контекста без потери разрешения
        self.backend = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)  # Выход: 1 канал (карта плотности)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return torch.relu(x)  # Плотность не может быть отрицательной


def train():
    if not os.path.exists(XML_PATH):
        print(f"❌ Файл '{XML_PATH}' не найден!")
        return

    samples = parse_cvat_xml(XML_PATH, IMAGES_DIR)
    print(f"📂 Найдено размеченных фото: {len(samples)}")
    if len(samples) == 0:
        print("❌ Не найдено совпадений между именами в XML и файлами в папке!")
        return

    # Делим на train (85%) и val (15%)
    np.random.seed(42)
    np.random.shuffle(samples)
    split = max(1, int(len(samples) * 0.85))
    train_samples = samples[:split]
    val_samples = samples[split:] if len(samples) > 1 else samples

    train_loader = DataLoader(
        PotatoDensityDataset(train_samples, IMG_SIZE, augment=True),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        PotatoDensityDataset(val_samples, IMG_SIZE, augment=False),
        batch_size=1, shuffle=False
    )

    device = get_compute_device()
    model = PotatoCSRNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction="sum")

    best_mae = float("inf")
    print("🚀 Начинаем обучение модели плотности...")

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

        # Валидация (считаем среднюю ошибку в штуках картофеля — MAE)
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                preds = model(imgs)
                pred_count = preds.sum().item()
                true_count = targets.sum().item()
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
