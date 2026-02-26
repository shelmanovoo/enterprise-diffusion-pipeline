# Enterprise Diffusion Pipeline

REST API для генерации изображений на базе Stable Diffusion XL.

---

## Описание

FastAPI-сервер с оптимизированным инференсом SDXL. Принимает текстовый промпт и возвращает PNG-изображение.

**Пример:** `POST /generate` с промптом «industrial robot, high detail, CAD style» → PNG.

---

## Архитектура

```
[Пользователь] → [SDXL API] → PNG
```

| Компонент | Технологии | Описание |
|-----------|------------|----------|
| **SDXL API** | FastAPI, Diffusers, PyTorch | REST API для генерации изображений по промпту |

---

## Структура проекта

```
3d/
├── server-prd_1.py      # FastAPI-сервер SDXL
├── download_sdxl.py     # Загрузка модели SDXL на USB/диск
├── requirements.txt     # Python-зависимости
└── README.md
```

---

## Требования

- **Python 3.10+** с поддержкой CUDA (рекомендуется для GPU)
- **Модель SDXL** — `stabilityai/stable-diffusion-xl-base-1.0` (≈6.5 GB)

---

## Установка и запуск

### 1. Загрузка модели SDXL

```bash
python download_sdxl.py
```

Скрипт ожидает смонтированный диск в `/mnt/usb`. Путь к модели в `server-prd_1.py` по умолчанию: `/mnt/usb/models/sdxl`. Измените `MODEL_PATH`, если модель лежит в другом месте.

### 2. SDXL API (сервер)

```bash
pip install -r requirements.txt
uvicorn server-prd_1:app --host 0.0.0.0 --port 5001
```

---

## API

### POST `/generate`

Генерация изображения по промпту.

**Request (JSON):**
```json
{
  "prompt": "industrial robot, high detail, CAD style, ...",
  "steps": 30,
  "width": 1024,
  "height": 1024,
  "seed": null
}
```

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `prompt` | string | — | Текстовый промпт (до 1000 символов) |
| `steps` | int | 30 | Количество шагов инференса (1–50) |
| `width` | int | 1024 | Ширина (256–1024) |
| `height` | int | 1024 | Высота (256–1024) |
| `seed` | int \| null | null | Seed для воспроизводимости |

**Response:** PNG-изображение (`image/png`)

### GET `/health`

Проверка состояния сервера: device (CPU/CUDA), доступность CUDA, занятая VRAM.

---

## Оптимизации сервера

- `attention_slicing` и `vae_slicing` — снижение потребления памяти
- `xformers` — эффективное внимание на GPU
- `torch.compile` (PyTorch 2+) — ускорение UNet
- Потокобезопасность через `Lock` для конкурентных запросов

---

## Лицензия

MIT
