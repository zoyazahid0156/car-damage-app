# Builds the car damage + part price server for Railway (CPU only).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Chrome for the PakWheels price scraper (installed below)
    CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER=/usr/bin/chromedriver \
    CHROME_NO_SANDBOX=1 \
    HEADLESS=1 \
    # Detection settings for a cloud CPU. Faster, slightly less sensitive. Raise them in Railway > Variables:
    #   DAMAGE_TTA=1  DAMAGE_TILES=2  (finds more damage, but 3-5x slower)
    DAMAGE_TTA=0 \
    DAMAGE_TILES=0

# Chrome + the libraries OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium chromium-driver libgl1 libglib2.0-0 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the normal one is ~2 GB bigger), then the rest
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-railway.txt .
RUN pip install -r requirements-railway.txt

COPY . .

# Railway gives the port in $PORT. Models are downloaded at start-up (fetch_models.py), then the server starts.
CMD ["sh", "-c", "python fetch_models.py && uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
