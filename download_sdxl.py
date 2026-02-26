#!/usr/bin/env python3
import os
import sys
from huggingface_hub import snapshot_download

# вставляем токен сюда
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Настройки ---
USB_MOUNT = "/mnt/usb"         # точка монтирования USB
MODEL_DIR = "models/sdxl"      # папка для модели
REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MAX_WORKERS = 1
ETAG_TIMEOUT = 600  # секунд для таймаута сети

# --- Проверка диска ---
if not os.path.ismount(USB_MOUNT):
    print(f"[ERROR] {USB_MOUNT} не смонтирован")
    sys.exit(1)

local_dir = os.path.join(USB_MOUNT, MODEL_DIR)
os.makedirs(local_dir, exist_ok=True)

if not os.access(local_dir, os.W_OK):
    print(f"[ERROR] Нет прав на запись в {local_dir}")
    print(f"Используй: sudo chown -R $USER:$USER {USB_MOUNT}")
    sys.exit(1)

# --- Включаем быстрый HF transfer ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Загрузка ---
print(f"[INFO] Начало загрузки модели {REPO_ID} в {local_dir}")

try:
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=local_dir,
        max_workers=MAX_WORKERS,
        etag_timeout=ETAG_TIMEOUT,
    )
except KeyboardInterrupt:
    print("\n[INFO] Загрузка прервана пользователем, можно продолжить повторным запуском скрипта")
except Exception as e:
    print(f"[ERROR] Произошла ошибка при скачивании: {e}")
    sys.exit(1)

print(f"[INFO] Загрузка завершена. Модель находится в {local_dir}")