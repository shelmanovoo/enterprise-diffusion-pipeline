# Enterprise Diffusion Pipeline

Пайплайн для генерации изображений: LLM (Ollama) улучшает текстовый запрос, Stable Diffusion XL генерирует картинку.

## Архитектура

```
[Пользователь] → [C# Client] → [Ollama (LLM)] → промпт → [SDXL API] → PNG
```

1. **C# Client** — оркестратор: принимает запрос, отправляет в Ollama, получает детализированный промпт, вызывает diffusion API, сохраняет результат.
2. **Ollama** — генерирует SDXL-промпты из короткого описания (например: «промышленный робот, CAD style»).
3. **SDXL API** (FastAPI) — сервер на PyTorch/Diffusers, генерирует изображения по промпту.

## Компоненты

| Компонент | Технологии | Описание |
|-----------|------------|----------|
| Client | C# .NET, HttpClient | Консольное приложение, связывает LLM и diffusion |
| LLM | Ollama, Llama 3 | Улучшение промптов под SDXL |
| Diffusion | FastAPI, Diffusers, SDXL | REST API для генерации изображений |

## Требования

- **Ollama** — запущен локально на `localhost:11434`
- **SDXL API** — запущен на `192.168.1.37:5001` (или измени URL в `pipeline.cs`)
- Модель SDXL — путь `/mnt/usb/models/sdxl` (или свой путь в `server-prd_1.py`)

## Запуск

### 1. SDXL API (сервер)

```bash
# Установка зависимостей
pip install fastapi uvicorn diffusers torch

# Запуск
uvicorn server-prd_1:app --host 0.0.0.0 --port 5001
```

### 2. Ollama

```bash
ollama serve
ollama pull llama3
```

### 3. C# Client

```bash
dotnet run
```

## API

### POST `/generate`

Генерация изображения по промпту.

**Request:**
```json
{
  "prompt": "industrial robot, high detail, CAD style, ...",
  "steps": 30,
  "width": 1024,
  "height": 1024,
  "seed": null
}
```

**Response:** PNG-изображение (`image/png`)

### GET `/health`

Проверка состояния сервера (device, CUDA, VRAM).

## Лицензия

MIT
