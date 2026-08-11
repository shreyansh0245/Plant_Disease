# Use a lightweight python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model artifacts
COPY ./app ./app
COPY ./artifacts/model ./artifacts/model

# Expose port 8000
EXPOSE 8000

# Start uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
