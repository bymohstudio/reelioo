# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Install system dependencies
# Added 'default-libmysqlclient-dev' to fix the mysqlclient build error
RUN apt-get update && apt-get install -y \
    libgomp1 \
    graphviz \
    pkg-config \
    libhdf5-dev \
    gcc \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. Set work directory
WORKDIR /app

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy project code
COPY . .

# 7. Collect static files
RUN python manage.py collectstatic --noinput

# 8. Start the application
CMD gunicorn reelioo.wsgi:application --workers 1 --threads 8 --timeout 120