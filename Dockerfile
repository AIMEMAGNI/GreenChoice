# Use Python 3.12.10 slim base image
FROM python:3.12.10-slim

# Environment variables for better behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PORT=8000 \
    PYTHONIOENCODING=utf-8

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Optional: for FastAPI live reload locally (ignored in prod)
ENV RELOAD="0"

# Expose port (for local use / doc clarity)
EXPOSE 8000

# Run with dynamic port for Render or Cloud Run
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
