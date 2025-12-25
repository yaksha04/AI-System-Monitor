FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    procps \
    sysstat \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p data/logs data/models data/metrics config

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Expose Streamlit port
EXPOSE 8501

# Health check (leave it)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psutil; psutil.cpu_percent()" || exit 1

# ⭐⭐ THE REAL FIX HERE ⭐⭐
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
