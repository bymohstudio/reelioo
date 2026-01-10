# ============================================================
# Python Base
# ============================================================
FROM python:3.13-slim

# ============================================================
# System deps (Python + Node for Tailwind)
# ============================================================
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Environment
# ============================================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ============================================================
# Python deps
# ============================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Tailwind deps
# ============================================================
COPY theme/static_src/package.json theme/static_src/package-lock.json ./theme/static_src/
RUN cd theme && npm install

# ============================================================
# App source
# ============================================================
COPY . .

# ============================================================
# Build Tailwind
# ============================================================
RUN python manage.py tailwind build

# ============================================================
# Collect static
# ============================================================
RUN python manage.py collectstatic --noinput

# ============================================================
# Run server
# ============================================================
CMD gunicorn reelioo.wsgi:application --bind 0.0.0.0:$PORT
