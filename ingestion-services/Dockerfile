# ingestion-service/Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install LibreOffice headless, fonts, and required build utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-nogui \
    default-jre-headless \
    fonts-liberation \
    fonts-dejavu-core \
    libgl1 \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
