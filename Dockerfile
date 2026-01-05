# 1. Use Python 3.13 Slim
FROM python:3.13-slim

# 2. Install ONLY essential build tools
# 'gcc' and 'python3-dev' are kept because some pip packages (like cffi) need them to compile.
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Set work directory
WORKDIR /app

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy project code
COPY . .

# 7. Collect static files (CSS/JS)
RUN python manage.py collectstatic --noinput

# 8. Start the application
CMD gunicorn reelioo.wsgi:application --bind 0.0.0.0:$PORT