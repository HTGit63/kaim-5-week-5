FROM python:3.9-slim

WORKDIR /app

# Copy only the lean API requirements
COPY requirements-api.txt .

# Give pip more time and retries for slow networks
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# Install all API deps in one shot
RUN pip install --no-cache-dir --disable-pip-version-check \
        --default-timeout=300 \
        --retries=5 \
        -r requirements-api.txt

# Copy your application code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
