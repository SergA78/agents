# =============================================================================
# Competitor Intelligence Agent — application image
# =============================================================================
FROM python:3.11-slim

# Не создавать .pyc, не буферизировать stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Системные зависимости (для psycopg / lxml и сборки колёс)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Сначала зависимости — для кэширования слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
