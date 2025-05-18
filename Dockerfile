# Base image
FROM python:3.9-slim as base

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend image
FROM base as frontend

WORKDIR /app/frontend

# Copy frontend code
COPY ./frontend ./

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Backend image
FROM base as backend

WORKDIR /app

# Copy backend code
COPY ./backend ./backend
COPY ./ml ./ml
COPY ./services ./services

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# MLflow image
FROM base as mlflow

WORKDIR /app

# Create mlruns directory
RUN mkdir -p /app/mlruns

# Expose MLflow port
EXPOSE 5000

# Run MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
