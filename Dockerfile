# 1. Use an official Python runtime
FROM python:3.11-slim

# 2. Install the MISSING system library (libgomp1)
# This command runs BEFORE Python starts, guaranteeing the library exists.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    graphviz \
    pkg-config \
    libhdf5-dev \
    gcc \
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
# IMPORTANT: Ensure 'reelioo.wsgi' matches your project folder name!
# If your main folder is 'core', change this to 'core.wsgi'
CMD gunicorn reelioo.wsgi:application --bind 0.0.0.0:$PORT